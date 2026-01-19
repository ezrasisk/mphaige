use nalgebra::{DMatrix, SVD};
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use tch::{Tensor, nn::VarStore};

pub struct Sheaf {
    dims: HashMap<usize, usize>,
    restrictions: HashMap<(usize, usize), Tensor>,  // Learnable tensors
    vs: VarStore,  // For training
}

impl Sheaf {
    pub fn new(vs: &VarStore, dims: HashMap<usize, usize>, edges: &[(usize, usize)], dim_restriction: usize) -> Self {
        let mut restrictions = HashMap::new();
        for &(u, v) in edges {
            let du = dims[&u];
            let dv = dims[&v];
            let path = vs.root().named_tensor(&format!("res_{}_{}", u, v), &[du as i64, dv as i64]);
            restrictions.insert((u, v), path);
        }
        Self { dims, restrictions, vs: vs.clone() }
    }

    pub fn build_laplacian(&self, g: &DiGraph<(), ()>, use_pinv: bool) -> Tensor {
        let total_dim = self.dims.values().sum::<usize>() as i64;
        let mut delta = Tensor::zeros(&[total_dim, total_dim], (tch::Kind::Float, tch::Device::Cpu));

        let mut offsets = HashMap::new();
        let mut current_offset = 0;
        for (&node, &d) in &self.dims {
            offsets.insert(node, current_offset);
            current_offset += d;
        }

        for edge in g.edge_indices() {
            let (u_node, v_node) = g.edge_endpoints(edge).unwrap();
            let u = u_node.index();
            let v = v_node.index();

            let phi = self.restrictions[&(u, v)].shallow_clone();
            let du = self.dims[&u] as i64;
            let dv = self.dims[&v] as i64;
            let ou = offsets[&u] as i64;
            let ov = offsets[&v] as i64;

            let phi_adj = if use_pinv {
                // Pseudo-inverse via SVD (convert to nalgebra, then back)
                let phi_mat = DMatrix::from_vec(du as usize, dv as usize, phi.flatten().f64_vec());
                let svd = SVD::new(phi_mat, true, true);
                let pinv_mat = svd.pseudo_inverse(1e-10).unwrap_or_else(|_| DMatrix::zeros(dv as usize, du as usize));
                Tensor::from_slice2(&pinv_mat.as_slice())
            } else {
                phi.t()
            };

            // u-diagonal +I
            let i_du = Tensor::eye(du, (tch::Kind::Float, tch::Device::Cpu));
            delta.slice(0, ou, ou + du, 1).slice(1, ou, ou + du, 1).add_(&i_du);

            // v-diagonal +phi_adj @ phi
            let contrib_vv = phi_adj.matmul(&phi);
            delta.slice(0, ov, ov + dv, 1).slice(1, ov, ov + dv, 1).add_(&contrib_vv);

            // ou-ov -phi
            delta.slice(0, ou, ou + du, 1).slice(1, ov, ov + dv, 1).sub_(&phi);

            // ov-ou -phi_adj.T
            let contrib_vu = phi_adj.t();
            delta.slice(0, ov, ov + dv, 1).slice(1, ou, ou + du, 1).sub_(&contrib_vu);
        }
        delta
    }

    // Diffusion: X_new = X - gamma * Delta @ X
    pub fn diffuse(&self, x: &Tensor, gamma: f64, g: &DiGraph<(), ()>, use_pinv: bool) -> Tensor {
        let delta = self.build_laplacian(g, use_pinv);
        x - gamma * delta.matmul(x)
    }

    // Cohomology-inspired regularization: Approx H^1 dim via trace( (Delta + eps I)^-1 )
    pub fn cohomology_reg(&self, g: &DiGraph<(), ()>, eps: f64) -> Tensor {
        let delta = self.build_laplacian(g, false);
        (delta + eps * Tensor::eye(delta.size()[0], (tch::Kind::Float, tch::Device::Cpu))).inverse().unwrap().trace()
    }
}

// Other utils: Sheaf attention, spectral gap, etc. (stub for now)
pub fn sheaf_attention(sheaf: &Sheaf, x: &Tensor) -> Tensor {
    // Eclectic: Softmax on restriction norms for dynamic weighting
    x.clone()  // Expand later
}

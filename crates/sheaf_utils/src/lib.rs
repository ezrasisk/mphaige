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

//more stuff
pub fn sheaf_attention(sheaf: &Sheaf, x: &Tensor, g: &DiGraph<(), ()>) -> Tensor {
    let mut attn_scores = Tensor::zeros(&[g.edge_count() as i64, 1], (tch::Kind::Float, tch::Device::Cpu));
    let mut idx = 0;
    for edge in g.edge_indices() {
        let (u_node, v_node) = g.edge_endpoints(edge).unwrap();
        let u = u_node.index();
        let v = v_node.index();
        let phi = sheaf.restrictions[&(u, v)].shallow_clone();
        let xu = x.slice(0, sheaf.offsets[&u] as i64, (sheaf.offsets[&u] + sheaf.dims[&u]) as i64, 1);
        let xv = x.slice(0, sheaf.offsets[&v] as i64, (sheaf.offsets[&v] + sheaf.dims[&v]) as i64, 1);
        let score = xu.matmul(&phi).dot(&xv);  // Simple dot-product attention
        attn_scores[idx] = score;
        idx += 1;
    }
    let attn_weights = attn_scores.softmax(0, tch::Kind::Float);
    // Apply weights to restrictions (eclectic: multiply phi by weight)
    let mut weighted_x = x.clone();
    idx = 0;
    for edge in g.edge_indices() {
        let (u_node, v_node) = g.edge_endpoints(edge).unwrap();
        let u = u_node.index();
        let v = v_node.index();
        let weight = attn_weights[idx].unsqueeze(0).unsqueeze(0);
        let phi_weighted = sheaf.restrictions[&(u, v)].mul(&weight);
        let contribution = phi_weighted.matmul(&weighted_x.slice(0, sheaf.offsets[&v] as i64, (sheaf.offsets[&v] + sheaf.dims[&v]) as i64, 1));
        weighted_x.slice(0, sheaf.offsets[&u] as i64, (sheaf.offsets[&u] + sheaf.dims[&u]) as i64, 1).add_(&contribution);
        idx += 1;
    }
    weighted_x
}

pub fn spectral_gap(delta: &Tensor) -> f64 {
    let eigenvalues = delta.symeig(false).1;  // Eigenvalues (ascending)
    eigenvalues[1].f64_value(&[]) - eigenvalues[0].f64_value(&[])  // Gap = λ2 - λ1 (λ1≈0)
}

pub fn cohomology_dim_approx(delta: &Tensor, eps: f64) -> f64 {
    // H^1 dim approx via rank of (Delta + eps I)^-1 trace
    let reg_delta = delta + eps * Tensor::eye(delta.size()[0], (tch::Kind::Float, tch::Device::Cpu));
    reg_delta.inverse().unwrap().trace().f64_value(&[])
}

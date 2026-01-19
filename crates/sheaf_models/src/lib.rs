use crate::sheaf_utils::{Sheaf, sheaf_attention};
use tch::{nn, Tensor, Device};

pub struct SNNModel {
    sheaf: Sheaf,
    layers: nn::Sequential,
}

impl SNNModel {
    pub fn new(vs: &nn::VarStore, dims: HashMap<usize, usize>, edges: &[(usize, usize)], num_layers: usize, hidden_dim: usize, out_dim: usize) -> Self {
        let sheaf = Sheaf::new(vs, dims, edges, hidden_dim);

        let mut layers = nn::seq();
        for _ in 0..num_layers {
            layers = layers.add(nn::linear(vs.root(), hidden_dim, hidden_dim, Default::default()));
        }
        layers = layers.add(nn::linear(vs.root(), hidden_dim, out_dim, Default::default()));

        Self { sheaf, layers }
    }

    pub fn forward(&self, x: &Tensor, g: &DiGraph<(), ()>, use_pinv: bool) -> Tensor {
        let mut x_out = self.sheaf.diffuse(x, 0.1, g, use_pinv);
        x_out = sheaf_attention(&self.sheaf, &x_out);
        self.layers.forward(&x_out)
    }

    pub fn train_step(&self, optimizer: &mut nn::Optimizer, loss_fn: fn(&Tensor, &Tensor) -> Tensor, x: &Tensor, y: &Tensor, g: &DiGraph<(), ()>) {
        let pred = self.forward(x, g, false);
        let loss = loss_fn(&pred, y) + 0.01 * self.sheaf.cohomology_reg(g, 1e-3);  // Optimal reg from research
        optimizer.backward_step(&loss);
    }
}

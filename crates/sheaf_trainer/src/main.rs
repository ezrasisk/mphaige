use clap::Parser;
use nalgebra::{DMatrix, SVD};
use petgraph::graph::DiGraph;
use rand::Rng;
use std::collections::HashMap;
use tch::{nn, Device, IndexOp, Tensor};
use sheaf_utils::{Sheaf, sheaf_attention, spectral_gap, cohomology_dim_approx};  // assume these are exported

#[derive(Parser, Debug)]
#[clap(version = "0.1.0", about = "Sheaf NN Trainer")]
struct Args {
    /// Path to JSON dataset file
    #[clap(long)]
    dataset: Option<String>,
}

fn main() {
    let args = Args::parse();
    println!("=== Sheaf Neural Network Trainer ===");
    println!("Dataset: {}\n", args.dataset.as_deref().unwrap_or("demo mode"));

    // Load or generate demo data
    let (g, dims, edges, features, labels) = if let Some(path) = args.dataset {
        load_dataset(&path)
    } else {
        generate_demo_data()
    };

    let total_dim = dims.values().sum::<usize>() as i64;
    println!("Total feature dim: {}", total_dim);
    println!("Offsets: {:?}", compute_offsets(&dims));

    // Device (GPU if available)
    let device = Device::CudaIfAvailable(0);
    let vs = nn::VarStore::new(device);

    // Model
    let mut model = SNNModel::new(&vs, dims, &edges, 3, 64, labels.size()[1] as usize);

    // Optimizer
    let mut opt = nn::Adam::default().build(&vs, 0.001).unwrap();

    // Training loop (short for demo)
    for epoch in 0..50 {
        opt.zero_grad();

        let pred = model.forward(&features, &g, true);  // true = use_pinv
        let loss_task = pred.cross_entropy_loss(&labels);

        let delta = model.sheaf.build_laplacian(&g, true);
        let reg_coh = 0.01 * cohomology_dim_approx(&delta, 1e-3);
        let reg_gap = -0.05 * spectral_gap(&delta);  // negative to maximize gap

        let total_loss = loss_task + reg_coh + reg_gap;

        total_loss.backward();
        opt.step();

        println!("Epoch {:3} | Loss: {:.4} | Task: {:.4} | CohReg: {:.4} | Gap: {:.4}",
                 epoch, total_loss.double_value(&[]),
                 loss_task.double_value(&[]),
                 reg_coh.double_value(&[]),
                 -reg_gap.double_value(&[]));
    }

    // Dummy evaluation (full split would go here)
    let test_pred = model.forward(&features, &g, false);
    let acc = (test_pred.argmax(1, false).eq1(&labels)).double_value(&[]).mean();
    println!("\nFinal accuracy (dummy full-set eval): {:.2}%", acc * 100.0);

    println!("\nTraining complete. Model saved in VarStore (add save/load as needed).");
}

// Stub: Replace with real JSON/CSV loader
fn load_dataset(path: &str) -> (DiGraph<(), ()>, HashMap<usize, usize>, Vec<(usize, usize)>, Tensor, Tensor) {
    println!("Loading dataset from: {}", path);
    // Placeholder - in real version: parse JSON → build graph, dims, edges, features, labels
    generate_demo_data()
}

// Demo data generator
fn generate_demo_data() -> (DiGraph<(), ()>, HashMap<usize, usize>, Vec<(usize, usize)>, Tensor, Tensor) {
    let mut g: DiGraph<(), ()> = DiGraph::new();
    let nodes = (0..6).map(|_| g.add_node(())).collect::<Vec<_>>();

    let edges = vec![(0,1), (1,2), (2,3), (0,4), (4,5), (3,5)];
    for &(u, v) in &edges {
        g.add_edge(nodes[u], nodes[v], ());
    }

    let mut dims: HashMap<usize, usize> = HashMap::new();
    dims.insert(0, 8);
    dims.insert(1, 16);
    dims.insert(2, 16);
    dims.insert(3, 8);
    dims.insert(4, 12);
    dims.insert(5, 8);

    let total_dim = dims.values().sum::<usize>() as i64;

    let mut rng = rand::thread_rng();
    let features_data: Vec<f32> = (0..(total_dim * 1)).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let features = Tensor::from_slice(&features_data).reshape(&[total_dim, 1]);

    let labels_data: Vec<i64> = vec![0, 1, 0, 1, 0, 1];  // dummy binary labels
    let labels = Tensor::from_slice(&labels_data).reshape(&[6, 1]);

    (g, dims, edges, features, labels)
}

fn compute_offsets(dims: &HashMap<usize, usize>) -> HashMap<usize, usize> {
    let mut offsets = HashMap::new();
    let mut offset = 0;
    for (&node, &d) in dims {
        offsets.insert(node, offset);
        offset += d;
    }
    offsets
}

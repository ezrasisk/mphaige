use sheaf_models::SNNModel;
use sheaf_utils::load_dataset;  // Stub: Load heterophily data from JSON/CSV
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    dataset: String,  // e.g., "cornell.json"
}

fn load_dataset(path: &str) -> (DiGraph<(), ()>, HashMap<usize, usize>, Vec<(usize, usize)>, Tensor, Tensor) {
    // Parse JSON: nodes/dims/edges/features/labels
    let data = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();  // Stub parse
    // Build g, dims, edges from data
    let features = Tensor::from_slice2(&data.features);  // n x f
    let labels = Tensor::from_slice(&data.labels);  // n x 1
    (g, dims, edges, features, labels)
}

fn main() {
    let args = Args::parse();
    let (g, dims, edges, features, labels) = load_dataset(&args.dataset);

    let vs = VarStore::new(tch::Device::Cuda(0));  // GPU if available
    let model = SNNModel::new(&vs, dims, &edges, 3, 64, labels.size()[1] as usize);

    let mut opt = nn::Adam::default().build(&vs, 0.001).unwrap();

    for epoch in 0..200 {
        let pred = model.forward(&features, &g, true);
        let loss = pred.cross_entropy_loss(&labels);
        let reg = 0.01 * model.sheaf.cohomology_reg(&g, 1e-3) - 0.1 * spectral_gap(&model.sheaf.build_laplacian(&g, false));  // Maximize gap
        let total_loss = loss + reg;

        opt.backward_step(&total_loss);
        println!("Epoch {} loss: {}", epoch, total_loss.f64_value(&[]));
    }

    // Eval: accuracy on test mask (add split)
    let test_pred = model.forward(&features, &g, false);
    let acc = (test_pred.argmax(1, false) == labels).f_mean(tch::Kind::Float).f64_value(&[]);
    println!("Test accuracy: {:.2}%", acc * 100.0);
}

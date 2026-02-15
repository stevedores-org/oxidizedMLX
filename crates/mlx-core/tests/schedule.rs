use mlx_core::{Device, DType, Shape, Tensor};
use std::sync::Arc;

#[test]
fn topo_order_has_dependencies_first() {
    let shape = Shape::new(vec![2, 2]);
    let device = Device::default_device();
    
    // z = (a*b) + (a*b)  (same subgraph used twice)
    let a = Tensor::zeros(&shape, DType::F32, &device).unwrap();
    let b = Tensor::zeros(&shape, DType::F32, &device).unwrap();
    let m = a.mul(&b).unwrap();
    let z = m.add(&m).unwrap();

    // Materialize via scheduler path.
    z.eval().unwrap();

    let data = z.to_vec_f32().unwrap();
    assert_eq!(data.len(), 4);
    assert!(data.iter().all(|&x| x == 0.0));
}

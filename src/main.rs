use nanograd::Tensor;
use nanograd::nn::activation::{ tanh, softmax, log_softmax };
use nanograd::nn::loss::categorical_cross_entropy;


fn main() {
    let data_a = vec![1.0, 2.0, 3.0, 9.0].into_boxed_slice();
    // let data_test_labels = vec![0.0, 1.0, 1.0, 0.0].into_boxed_slice();
    // let data_b = vec![0.1, 0.2, 0.3, 0.4].into_boxed_slice();
    // let data_d = vec![5.0, 6.0, 7.0, 8.0].into_boxed_slice();
    // let dim_a = (2, 2);
    // let dim_b = (2, 2);
    // let tensor_a = Tensor::new(data_a, dim_a, None, Some(true));
    // let tensor_b = Tensor::new(data_b, dim_b, None, Some(true));
    // let tensor_c = tensor_a * tensor_b;
    // println!("{:}", tensor_c);
    // let tensor_d = Tensor::new(data_d, dim_a, None, Some(true));
    // let tensor_e = tensor_c + tensor_d;
    // let tensor_f: Tensor<f64> = Tensor::new(
    //     vec![11.0, 12.0, 13.0, 14.0].into_boxed_slice(),
    //     dim_a,
    //     None,
    //     Some(true)
    // );
    // let tensor_g = tensor_f * tensor_e;
    // // relu test
    // let mut tensor_h = relu(tensor_g);
    // tensor_h.backward();
    // println!("{:}", tensor_h);
    let tensor_a = Tensor::new(data_a, (2, 2), None, Some(true));
    // let tensor_b = Tensor::new(data_test_labels, (2, 2), None, Some(true));
    // let tensor_result = categorical_cross_entropy(tensor_a, tensor_b);
    // println!("{:}", tensor_result);
}

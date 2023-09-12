use crate::{ Tensor, TensorTrait, nn::activation::log_softmax };

/// Categorical cross entropy loss function
///
/// # Arguments
///
/// * `y_pred` - The predicted tensor
/// * `y_true` - The true tensor
///
/// # Returns
///
/// A  1x1 tensor with the categorical cross entropy loss
///
/// # Examples
///
/// ```
/// use nanograd::{ Tensor, nn::loss::categorical_cross_entropy };
///
/// let data_pred = vec![1.0, 2.0, 3.0, 9.0].into_boxed_slice();
/// let data_test_labels = vec![0.0, 1.0, 1.0, 0.0].into_boxed_slice();
/// let tensor_pred = Tensor::new(data_pred, (2, 2), None, Some(true));
/// let tensor_test = Tensor::new(data_test_labels, (2, 2), None, Some(true));
///
/// let tensor_result = categorical_cross_entropy(tensor_pred, tensor_test);
///
/// let expected_result = &vec![3.1578685995332485].into_boxed_slice();
///
/// assert_eq!(tensor_result.data(), expected_result);
///
pub fn categorical_cross_entropy<T: TensorTrait<T>>(
    y_pred: Tensor<T>,
    mut y_true: Tensor<T>
) -> Tensor<T> {
    // assert that y_pred and y_true have the same shape
    assert_eq!(y_pred.dim(), y_true.dim());
    // get the data of the true tensor
    let mut transformed_pred = log_softmax(y_pred);
    // total number of categories
    let num_categories: f64 = y_true.dim().1 as f64;
    assert!(num_categories > 0.0);
    // convert to generic t type and get the inverse
    // we use the inverse so we can express the final result
    // as a product of the sum and the inverse of the number of categories
    let num_categories_inv: T = T::from_f64(-1.0 / num_categories).unwrap();
    assert!(num_categories_inv != T::zero());
    // flatten the tensors
    transformed_pred.flatten();
    y_true.flatten();
    // transpose to fulfill the matrix multiplication dimension requirements
    y_true.transpose();
    let output = transformed_pred * y_true;
    // output should be a 1x1 tensor
    assert!(output.dim() == (1, 1));
    output * num_categories_inv
}

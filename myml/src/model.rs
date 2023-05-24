use std::io::{self, Write};
use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use crate::traits::{Layer, Loss};

pub struct Model<DI, N, Y, L> {
    input_dimension: PhantomData<DI>,
    training_data_type: PhantomData<Y>,
    net: N,
    params: Array<f32, Ix1>,
    loss: L,
    learning_rate: f32,
}

impl<DI, N, Y, L> Model<DI, N, Y, L>
where
    DI: Dimension,
    N: Layer<DI>,
    L: Loss<N::Output, Y>,
    Y: Copy,
{
    pub fn new(net: N, loss: L) -> Self {
        let n = net.num_params();
        Model {
            input_dimension: PhantomData,
            training_data_type: PhantomData,
            net,
            params: 0.1f32 * Array::random((n,), StandardNormal),
            loss,
            learning_rate: 0.1,
        }
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }

    /// Run the model on an array of examples.
    pub fn apply(&self, x: ArrayView<'_, f32, DI>) -> Array<f32, N::Output> {
        let input_shape = x.raw_dim();
        let output_shape = self.net.output_shape(input_shape.clone());
        let mut tmp = Array1::<f32>::zeros(self.net.num_hidden_activations(input_shape));
        let mut out = Array::<f32, N::Output>::zeros(output_shape);
        self.net
            .apply(self.params.view(), x, tmp.view_mut(), out.view_mut());
        out
    }

    /// Train the model on a batch of examples.
    pub fn train(&mut self, x_train: ArrayView<'_, f32, DI>, y_train: Y, rate: f32) -> (f32, f32) {
        let input_shape = x_train.raw_dim();
        let output_shape = self.net.output_shape(input_shape.clone());
        let mut tmp = Array1::<f32>::zeros(self.net.num_hidden_activations(input_shape));
        let mut yh = Array::<f32, N::Output>::zeros(output_shape);
        self.net.apply(
            self.params.view(),
            x_train.view(),
            tmp.view_mut(),
            yh.view_mut(),
        );
        let loss = self.loss.loss(y_train, yh.view());
        let accuracy = self.loss.accuracy(y_train, yh.view());

        let dyh = self.loss.deriv(y_train, yh.view());
        let mut dp = Array1::<f32>::zeros(self.net.num_params());
        let _ = self.net.derivatives(
            self.params.view(),
            x_train,
            tmp.view(),
            dyh.view(),
            dp.view_mut(),
        );
        if dp.iter().all(|x| *x == 0.0) {
            println!("gradient is 0");
            return (0.0, 1.0);
        }

        self.params -= &(rate * dp);
        (loss, accuracy)
    }

    /// Train the model on a data set `epochs`.
    pub fn train_epochs<'e, I, E>(&mut self, epochs: I)
    where
        I: IntoIterator<Item = E>,
        E: IntoIterator<Item = (ArrayView<'e, f32, DI>, Y)>,
    {
        let epochs: Vec<E> = epochs.into_iter().collect();
        let num_epochs = epochs.len();
        let mut pairs_per_epoch = 0;
        let mut progress = 0.0;

        for (i, epoch) in epochs.into_iter().enumerate() {
            print!("epoch {i} - \x1b[s");
            io::stdout().flush().unwrap();

            let mut n_total = 0;
            let mut loss_total = 0.0;
            let mut accuracy_total = 0.0;

            for (x, y) in epoch {
                let n = x.len();
                n_total += n;
                if i > 0 {
                    let batch_progress =
                        n_total.min(pairs_per_epoch) as f32 / pairs_per_epoch as f32;
                    progress = ((i - 1) as f32 + batch_progress) / (num_epochs - 1) as f32;
                }
                let (last_loss, last_accuracy) =
                    self.train(x.view(), y, self.learning_rate * (1.0 - progress));

                if n > 0 {
                    loss_total += last_loss * n as f32;
                    accuracy_total += last_accuracy * n as f32;
                    let loss = loss_total / n_total as f32;
                    let accuracy = accuracy_total / n_total as f32;
                    print!(
                        "\x1b[u{}\x1b[u{} loss={loss:.4} accuracy={accuracy:.4}",
                        " ".repeat(78),
                        Self::progress_bar(progress, 40)
                    );
                    io::stdout().flush().unwrap();
                }
            }
            if i == 0 {
                pairs_per_epoch = n_total;
            }

            println!();
        }
    }

    fn progress_bar(progress: f32, len: usize) -> String {
        let n = (len as f32 * progress.min(1.0)) as usize;
        return format!("[{}{}]", "#".repeat(n), " ".repeat(len - n));
    }
}

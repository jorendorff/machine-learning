use ndarray::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use crate::traits::{Layer, Loss};

pub struct Model<DI, DO, Y> {
    root: Box<dyn Layer<DI, DO>>,
    params: Array<f32, Ix1>,
    loss: Box<dyn Loss<DO, Y>>,
    learning_rate: f32,
}

impl<DI, DO, Y> Model<DI, DO, Y>
where
    DI: Dimension,
    DO: Dimension,
{
    pub fn new(root: Box<dyn Layer<DI, DO>>, loss: Box<dyn Loss<DO, Y>>) -> Self {
        let n = root.num_params();
        Model {
            root,
            params: 0.1f32 * Array::random((n,), StandardNormal),
            loss,
            learning_rate: 0.1,
        }
    }

    pub fn apply(&self, x: &Array<f32, DI>) -> Array<f32, DO> {
        self.root.apply(&self.params, &x)
    }

    pub fn train(&mut self, x_train: &Array<f32, DI>, y_train: &Y, rate: f32) -> (f32, f32) {
        let yh = self.apply(x_train);
        let loss = self.loss.loss(y_train, &yh);
        let accuracy = self.loss.accuracy(y_train, &yh);

        let dyh = self.loss.deriv(y_train, &yh);
        let mut dp = Array::<f32, Ix1>::zeros((self.root.num_params(),).f());
        let _ = self.root.derivatives(&self.params, &x_train, &dyh, &mut dp);
        if dp.iter().all(|x| *x == 0.0) {
            println!("gradient is 0");
            return (0.0, 1.0);
        }

        self.params -= &(rate * dp);
        (loss, accuracy)
    }

    pub fn train_epochs<I, E>(&mut self, epochs: I)
    where
        I: IntoIterator<Item = E>,
        E: IntoIterator<Item = (Array<f32, DI>, Y)>,
    {
        let epochs: Vec<E> = epochs.into_iter().collect();
        let num_epochs = epochs.len();
        let mut pairs_per_epoch = 0;
        let mut progress = 0.0;

        for (i, epoch) in epochs.into_iter().enumerate() {
            print!("epoch {i} - \x1b[s");

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
                    self.train(&x, &y, self.learning_rate * (1.0 - progress));

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

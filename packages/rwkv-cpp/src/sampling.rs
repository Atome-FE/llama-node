// a lot of codes borrowed from https://github.com/KerfuffleV2/smolrsrwkv/blob/main/smolrwkv/src/util.rs
use std::ops::{Add, Sub};

use ndarray::{Array1, ArrayView1, NdFloat, ScalarOperand};
use num_traits::FromPrimitive;
use rand::{rngs::StdRng, SeedableRng};

pub trait ReqOps: Sized + Default + Clone
where
    Self: NdFloat + ScalarOperand + FromPrimitive,
    Self: for<'a> Sub<&'a Array1<Self>, Output = Array1<Self>>,
    Self: for<'a> Add<&'a Array1<Self>, Output = Array1<Self>>,
{
}

impl ReqOps for f32 {}
impl ReqOps for f64 {}

pub fn softmax<T: ReqOps>(x: &ArrayView1<T>) -> Array1<T> {
    let x_exp = x.mapv(T::exp);
    &x_exp / x_exp.sum()
}

pub fn sample_logits(logits: &mut [f32], temp: f32, top_p: f32, seed: &Option<u64>) -> usize {
    let binding = Array1::from(logits.to_vec());
    let logits = binding.view();
    let probs = softmax(&logits);
    sample_probs(&probs, temp, top_p, seed)
}

pub fn sample_probs<T: ReqOps + num_traits::AsPrimitive<f32>>(
    probs: &Array1<T>,
    temp: f32,
    mut top_p: f32,
    seed: &Option<u64>,
) -> usize {
    use rand::distributions::{Distribution, WeightedError, WeightedIndex};

    let mut rng: rand::rngs::StdRng = if let Some(seed) = seed {
        StdRng::seed_from_u64(*seed)
    } else {
        StdRng::from_entropy()
    };

    // const EOT_TOKEN_ID: usize = 0;

    if top_p == 0.0 {
        top_p = 1.0;
    }

    // if temp == 0.0 {
    //     // get argmax of probs
    //     let index_of_max = probs
    //         .iter()
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //         .map(|(i, _)| i)
    //         .unwrap_or(0);
    //     return index_of_max;
    // }

    // if top_p < 1.0 {
    // sort the probs
    let mut sorted_probs = probs.to_vec();
    sorted_probs.sort_by(|a, b| {
        T::partial_cmp(a, b)
            .unwrap_or(std::cmp::Ordering::Greater)
            .reverse()
    });
    let mut cumulative_probs = Vec::with_capacity(sorted_probs.len());
    let _ = sorted_probs.iter().fold(T::zero(), |acc, i| {
        let newcum = acc + *i;
        cumulative_probs.push(newcum);
        newcum
    });

    let cutoffidx = cumulative_probs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| v.as_() > top_p)
        .map(|i| i.0)
        .unwrap_or_default();

    let cutoff = sorted_probs[cutoffidx].as_();

    let probs = probs.map(|i| {
        let i: f32 = i.as_();
        if i < cutoff {
            0.0
        } else {
            i
        }
    });

    let probs = &probs / probs.sum();
    let dist = match WeightedIndex::new(probs.iter().map(|val| val.powf(1.0 / temp))) {
        Ok(dist) => dist,
        Err(WeightedError::AllWeightsZero) => {
            return 0;
        }
        e => e.expect("Bad weight"),
    };
    dist.sample(&mut rng)
}

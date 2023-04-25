use rand::Rng;

pub fn random_choice(probs: &[f32]) -> usize {
    // Generate a random number between 0 and 1
    let random_number = rand::thread_rng().gen_range(0.0..1.0);

    // Iterate through the probabilities to find the corresponding outcome
    let mut cumulative_prob = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative_prob += prob;
        if random_number < cumulative_prob {
            return i;
        }
    }

    // If no outcome is chosen, return the last index
    probs.len() - 1
}

pub fn soft_max(arr: &[f32]) -> Vec<f32> {
    let exp = arr.iter().map(|x| x.exp()).collect::<Vec<f32>>();
    let sum_exp = exp.iter().sum::<f32>();
    exp.iter().map(|x| x / sum_exp).collect::<Vec<f32>>()
}

pub fn sample_logits(logits: &mut [f32], temp: f32, top_p: f32) -> usize {
    // println!("logits: {:?}", logits);
    let logits = logits.to_vec();
    // softmax on logits
    let probs = soft_max(&logits);
    sample_probs(&probs, temp, top_p)
}

pub fn sample_probs(probs: &[f32], temp: f32, mut top_p: f32) -> usize {
    // let probs = Array1::from(probs);
    let mut probs = probs.to_vec();
    if top_p == 0.0 {
        top_p = 1.0;
    }

    if temp == 0.0 {
        // get argmax of probs
        let index_of_max = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        return index_of_max;
    }

    if top_p < 1.0 {
        // sort the probs
        let mut sorted_probs = probs.clone();
        sorted_probs.sort_by(|a, b| a.total_cmp(b));
        sorted_probs.reverse();

        let cumulative_propbs = sorted_probs.iter().scan(0.0, |acc, x| {
            *acc += x;
            Some(*acc)
        });

        let cutoff_index = cumulative_propbs
            .enumerate()
            .find(|(_, x)| *x > top_p)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let cutoff = sorted_probs[cutoff_index];

        probs = probs
            .iter()
            .map(|x| if *x < cutoff { 0.0 } else { *x })
            .collect::<Vec<f32>>();
    }

    if temp != 1.0 {
        probs = probs.iter().map(|x| x.powf(temp)).collect::<Vec<f32>>();
    }

    let sum: f32 = probs.iter().sum();
    probs = probs.iter().map(|x| x / sum).collect::<Vec<f32>>();

    random_choice(probs.as_slice())
}

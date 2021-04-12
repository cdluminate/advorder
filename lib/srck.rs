/* Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
 * Released under the Apache-2.0 License.
 *
 * Short-range Ranking Correlation Kernal (SRCK) [Rust Version]
 */
#![crate_type = "dylib"]
//use rayon::prelude::*;
use std::slice;
use std::collections::HashMap;
//use std::ops::Add;

// Constants
const REWARD_CONCORDANT: i32 = 1;
const REWARD_DISCORDANT: i32 = -1;
const REWARD_OUTOFRANGE: i32 = -1;
// experimental feature: add an extra semantics preserving penalty. (not included in paper)
const EXTRA_SEM_PRES: bool = false;

#[repr(C)]
pub struct array {
    data: *const i32,
    len: usize,
}

#[no_mangle]
pub extern "C" fn print_array(arx: array) {
    // print an array: just for debugging
    let x = array2slice(arx);
    print!("[");
    for i in 0..x.len() {
        if x.len() - 1 == i {
            print!("{}", x[i]);
        } else {
            print!("{} ", x[i]);
        }
    }
    println!("]");
}

fn array2slice(arx: array) -> &'static[i32] {
    // Convert a raw C array to a slice.
    unsafe {
        assert!(!arx.data.is_null());
        slice::from_raw_parts(arx.data, arx.len)
    }
}

#[no_mangle]
pub extern "C" fn ShortrangeRankingCorrelation(
    argsort: array, otopk: array, rperm: array) -> f32 {
    /* Short-range Ranking Correlation
     *
     * argsort: the actual ranking list, a list of candidate IDs.
     * otopk: original top-k result, a list of candidate IDs.
     * rperm: specified permutation, a shuffled range(0, len(otopk)).
     */
    // create constants
    let rpermlen: usize = rperm.len;
    let nscore: f32 = ((rpermlen * (rpermlen - 1))/2) as f32;
    // convertion from raw C array into slices
    let argsort: &[i32] = array2slice(argsort);
    let otopk: &[i32] = array2slice(otopk);
    let rperm: &[i32] = array2slice(rperm);
    // av2ij: id<int> -> rank<int> in the actual ranking list
    let mut av2ij: HashMap<i32, usize> = HashMap::new();
    argsort.iter().enumerate().for_each(|(i, &v)| {av2ij.insert(v, i); ()});
    // rtopk: permuted otopk by permutation vector (rperm).
    let mut rtopk: Vec<i32> = Vec::new();
    rperm.iter().for_each(|&i| {rtopk.push(otopk[i as usize]); ()});
    // rv2ij: id<int> -> rank<int> in the specified ranking list
    let mut rv2ij: HashMap<i32, usize> = HashMap::new();
    rtopk.iter().enumerate().for_each(|(i, &v)| {rv2ij.insert(v, i); ()});
    // scores
    let mut scores: Vec<i32> = Vec::new();
    for i in 0..rpermlen {
        // i-th original
        let io: i32 = otopk[i as usize];
        // out-of-range
        if let None = av2ij.get(&io) {
            (0..i).for_each(|_| scores.push(REWARD_OUTOFRANGE));
            continue;
        }
        for j in 0..i {
            // j-th original
            let jo: i32 = otopk[j as usize];
            // out-of-range
            if let None = av2ij.get(&jo) {
                scores.push(REWARD_OUTOFRANGE);
                continue;
            }
            // compare rank
            let cilj: bool =
                av2ij.get(&io).unwrap() < av2ij.get(&jo).unwrap();
            let xilj: bool =
                rv2ij.get(&io).unwrap() < rv2ij.get(&jo).unwrap();
            if cilj ^ xilj {
                // discordant
                scores.push(REWARD_DISCORDANT);
            } else {
                // concordant
                scores.push(REWARD_CONCORDANT);
            }
        }
    }
    // extra: semantics preserving (experimental)
    if EXTRA_SEM_PRES {
        let mut ov2ij: HashMap<i32, usize> = HashMap::new();
        otopk.iter().enumerate().for_each(
            |(i, &v)| {ov2ij.insert(v, i); ()});
        let cansee: i32 = 50; // only works in (*, 50, *) setting.
        let mut sp: Vec<f32> = Vec::new();
        for i in 0..rpermlen {
            match av2ij.get(&otopk[i as usize]) {
                Some(&curid) => {
                    let curid = curid as i32;
                    let origid: i32 = i as i32;
                    if curid - origid <= 0 {
                        continue;
                    } else {
                        sp.push(-((curid - origid) as f32)
                                /((cansee - origid) as f32));
                    }
                },
                None => continue,
            }
        }
        return (scores.iter().sum::<i32>() as f32)/nscore
            + sp.iter().sum::<f32>();
    }
    // sum and take the average.
    return (scores.iter().sum::<i32>() as f32) / nscore; // fast enough.
}

/* Work-in-progress */
/*
#[no_mangle]
pub extern "C" fn BatchedShortrangeRankingCorrelation(
    argsorts: &[array], otopk: array, rperm: array)
{
    let taus: Vec<f32> = (0..argsorts.len()).into_iter()
        .map(|i| ShortrangeRankingCorrelation(argsorts[i], otopk, rperm))
        .collect();
}
*/

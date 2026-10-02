//! Binary entry point. Everything lives in the library crate so the unit
//! tests can reach it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    wispr_local_lib::run();
}

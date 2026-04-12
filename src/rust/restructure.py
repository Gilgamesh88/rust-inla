import os, sys, shutil, re

base = 'C:/Users/Antonio/.gemini/antigravity/scratch/rustyINLA/src/rust'
core = os.path.join(base, 'inla_core')
os.makedirs(os.path.join(core, 'src'), exist_ok=True)
os.makedirs(os.path.join(core, 'tests'), exist_ok=True)

modules = ['error', 'graph', 'inference', 'likelihood', 'models', 'optimizer', 'problem', 'solver']
for mod in modules:
    src_path = os.path.join(base, 'src', mod)
    dst_path = os.path.join(core, 'src', mod)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

for f in ['utils.rs', 'config.rs', 'distributions.rs']:
    src_f = os.path.join(base, 'src', f)
    if os.path.exists(src_f):
        shutil.move(src_f, os.path.join(core, 'src', f))

core_lib = """pub mod error;
pub mod graph;
pub mod inference;
pub mod likelihood;
pub mod models;
pub mod optimizer;
pub mod problem;
pub mod solver;
"""
with open(os.path.join(core, 'src', 'lib.rs'), 'w', encoding='utf-8') as f:
    f.write(core_lib)

core_cargo = """[package]
name = "inla_core"
version = "0.2.0"
edition = "2021"

[dependencies]
faer = "0.24.0"
rayon = "1.10"
argmin = "0.10"
argmin-math = "0.4"
statrs = "0.18"
sha2 = "0.10"
rand = "0.8"
thiserror = "2.0"
ndarray = { version = "0.16", features = ["approx"] }
rustc-hash = "2.0"

[dev-dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
approx = "0.5"
criterion = { version = "0.5", features = ["html_reports"] }
"""
with open(os.path.join(core, 'Cargo.toml'), 'w', encoding='utf-8') as f:
    f.write(core_cargo)

main_cargo = """[package]
name = 'rustyINLA'
publish = false
version = '0.2.0'
edition = '2021'
rust-version = '1.65'

[lib]
crate-type = [ 'staticlib', 'rlib' ]
name = 'rustyINLA'

[workspace]
members = [ ".", "inla_core" ]

[dependencies]
extendr-api = { git = "https://github.com/extendr/extendr" }
inla_core = { path = "inla_core" }
rayon = "1.10"
rustc-hash = "2.0"
"""
with open(os.path.join(base, 'Cargo.toml'), 'w', encoding='utf-8') as f:
    f.write(main_cargo)

lib_rs_path = os.path.join(base, 'src', 'lib.rs')
with open(lib_rs_path, 'r', encoding='utf-8') as f:
    original_lib = f.read()

cleaned_lib = re.sub(r'pub mod error;.*?pub mod solver;\n', '', original_lib, flags=re.DOTALL)
cleaned_lib = "use inla_core::*;\n" + cleaned_lib
with open(lib_rs_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_lib)

print('Workspace Restructured!')

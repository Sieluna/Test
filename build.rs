use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    SpirvBuilder::new("kernels", "spirv-unknown-vulkan1.1")
        .print_metadata(MetadataPrintout::Full)
        .build()
        .expect("Kernel failed to compile");
}

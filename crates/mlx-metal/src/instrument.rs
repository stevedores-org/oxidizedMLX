/// Telemetry snapshot for a unified buffer, used to verify zero-copy behavior.
#[derive(Debug, Clone)]
pub struct BufferTelemetry {
    /// Address of the host allocation (Rust-side pointer).
    pub host_ptr: usize,
    /// Address returned by `MTLBuffer::contents()`.
    pub gpu_contents_ptr: usize,
    /// Bytes copied during buffer creation (0 for true no-copy).
    pub bytes_copied: usize,
    /// Logical byte length of the buffer.
    pub byte_length: usize,
    /// Whether the buffer was created via the no-copy path.
    pub is_no_copy: bool,
}

impl BufferTelemetry {
    /// Panics if this buffer was not truly zero-copy.
    pub fn assert_zero_copy(&self) {
        assert_eq!(
            self.host_ptr, self.gpu_contents_ptr,
            "host_ptr ({:#x}) != gpu_contents_ptr ({:#x}): Metal copied the data",
            self.host_ptr, self.gpu_contents_ptr,
        );
        assert_eq!(
            self.bytes_copied, 0,
            "bytes_copied was {}, expected 0 for zero-copy",
            self.bytes_copied,
        );
    }
}

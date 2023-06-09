/* automatically generated by rust-bindgen 0.64.0 */

pub const _VCRT_COMPILER_PREPROCESSOR: u32 = 1;
pub const _SAL_VERSION: u32 = 20;
pub const __SAL_H_VERSION: u32 = 180000000;
pub const _USE_DECLSPECS_FOR_SAL: u32 = 0;
pub const _USE_ATTRIBUTES_FOR_SAL: u32 = 0;
pub const _CRT_PACKING: u32 = 8;
pub const _HAS_EXCEPTIONS: u32 = 1;
pub const _STL_LANG: u32 = 0;
pub const _HAS_CXX17: u32 = 0;
pub const _HAS_CXX20: u32 = 0;
pub const _HAS_CXX23: u32 = 0;
pub const _HAS_NODISCARD: u32 = 0;
pub const WCHAR_MIN: u32 = 0;
pub const WCHAR_MAX: u32 = 65535;
pub const WINT_MIN: u32 = 0;
pub const WINT_MAX: u32 = 65535;
pub const __bool_true_false_are_defined: u32 = 1;
pub const true_: u32 = 1;
pub const false_: u32 = 0;
pub const RWKV_FILE_MAGIC: u32 = 1734831462;
pub const RWKV_FILE_VERSION_0: u32 = 100;
pub const RWKV_FILE_VERSION_1: u32 = 101;
pub const RWKV_FILE_VERSION_MIN: u32 = 100;
pub const RWKV_FILE_VERSION_MAX: u32 = 101;
pub const RWKV_FILE_VERSION: u32 = 101;
pub type wchar_t = ::std::os::raw::c_ushort;
pub type max_align_t = f64;
pub type va_list = *mut ::std::os::raw::c_char;
extern "C" {
    pub fn __va_start(arg1: *mut *mut ::std::os::raw::c_char, ...);
}
pub type __vcrt_bool = bool;
extern "C" {
    pub fn __security_init_cookie();
}
extern "C" {
    pub fn __security_check_cookie(_StackCookie: usize);
}
extern "C" {
    pub fn __report_gsfailure(_StackCookie: usize) -> !;
}
extern "C" {
    pub static mut __security_cookie: usize;
}
pub type int_least8_t = ::std::os::raw::c_schar;
pub type int_least16_t = ::std::os::raw::c_short;
pub type int_least32_t = ::std::os::raw::c_int;
pub type int_least64_t = ::std::os::raw::c_longlong;
pub type uint_least8_t = ::std::os::raw::c_uchar;
pub type uint_least16_t = ::std::os::raw::c_ushort;
pub type uint_least32_t = ::std::os::raw::c_uint;
pub type uint_least64_t = ::std::os::raw::c_ulonglong;
pub type int_fast8_t = ::std::os::raw::c_schar;
pub type int_fast16_t = ::std::os::raw::c_int;
pub type int_fast32_t = ::std::os::raw::c_int;
pub type int_fast64_t = ::std::os::raw::c_longlong;
pub type uint_fast8_t = ::std::os::raw::c_uchar;
pub type uint_fast16_t = ::std::os::raw::c_uint;
pub type uint_fast32_t = ::std::os::raw::c_uint;
pub type uint_fast64_t = ::std::os::raw::c_ulonglong;
pub type intmax_t = ::std::os::raw::c_longlong;
pub type uintmax_t = ::std::os::raw::c_ulonglong;
pub const rwkv_error_flags_RWKV_ERROR_NONE: rwkv_error_flags = 0;
pub const rwkv_error_flags_RWKV_ERROR_ARGS: rwkv_error_flags = 256;
pub const rwkv_error_flags_RWKV_ERROR_FILE: rwkv_error_flags = 512;
pub const rwkv_error_flags_RWKV_ERROR_MODEL: rwkv_error_flags = 768;
pub const rwkv_error_flags_RWKV_ERROR_MODEL_PARAMS: rwkv_error_flags = 1024;
pub const rwkv_error_flags_RWKV_ERROR_GRAPH: rwkv_error_flags = 1280;
pub const rwkv_error_flags_RWKV_ERROR_CTX: rwkv_error_flags = 1536;
pub const rwkv_error_flags_RWKV_ERROR_ALLOC: rwkv_error_flags = 1;
pub const rwkv_error_flags_RWKV_ERROR_FILE_OPEN: rwkv_error_flags = 2;
pub const rwkv_error_flags_RWKV_ERROR_FILE_STAT: rwkv_error_flags = 3;
pub const rwkv_error_flags_RWKV_ERROR_FILE_READ: rwkv_error_flags = 4;
pub const rwkv_error_flags_RWKV_ERROR_FILE_WRITE: rwkv_error_flags = 5;
pub const rwkv_error_flags_RWKV_ERROR_FILE_MAGIC: rwkv_error_flags = 6;
pub const rwkv_error_flags_RWKV_ERROR_FILE_VERSION: rwkv_error_flags = 7;
pub const rwkv_error_flags_RWKV_ERROR_DATA_TYPE: rwkv_error_flags = 8;
pub const rwkv_error_flags_RWKV_ERROR_UNSUPPORTED: rwkv_error_flags = 9;
pub const rwkv_error_flags_RWKV_ERROR_SHAPE: rwkv_error_flags = 10;
pub const rwkv_error_flags_RWKV_ERROR_DIMENSION: rwkv_error_flags = 11;
pub const rwkv_error_flags_RWKV_ERROR_KEY: rwkv_error_flags = 12;
pub const rwkv_error_flags_RWKV_ERROR_DATA: rwkv_error_flags = 13;
pub const rwkv_error_flags_RWKV_ERROR_PARAM_MISSING: rwkv_error_flags = 14;
pub type rwkv_error_flags = ::std::os::raw::c_int;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rwkv_context {
    _unused: [u8; 0],
}
extern "C" {
    pub fn rwkv_set_print_errors(ctx: *mut rwkv_context, print_errors: bool);
}
extern "C" {
    pub fn rwkv_get_print_errors(ctx: *mut rwkv_context) -> bool;
}
extern "C" {
    pub fn rwkv_get_last_error(ctx: *mut rwkv_context) -> rwkv_error_flags;
}
extern "C" {
    pub fn rwkv_init_from_file(
        model_file_path: *const ::std::os::raw::c_char,
        n_threads: u32,
    ) -> *mut rwkv_context;
}
extern "C" {
    pub fn rwkv_gpu_offload_layers(ctx: *const rwkv_context, n_gpu_layers: u32) -> bool;
}
extern "C" {
    pub fn rwkv_eval(
        ctx: *const rwkv_context,
        token: u32,
        state_in: *const f32,
        state_out: *mut f32,
        logits_out: *mut f32,
    ) -> bool;
}
extern "C" {
    pub fn rwkv_get_state_buffer_element_count(ctx: *const rwkv_context) -> u32;
}
extern "C" {
    pub fn rwkv_get_logits_buffer_element_count(ctx: *const rwkv_context) -> u32;
}
extern "C" {
    pub fn rwkv_free(ctx: *mut rwkv_context);
}
extern "C" {
    pub fn rwkv_quantize_model_file(
        model_file_path_in: *const ::std::os::raw::c_char,
        model_file_path_out: *const ::std::os::raw::c_char,
        format_name: *const ::std::os::raw::c_char,
    ) -> bool;
}
extern "C" {
    pub fn rwkv_get_system_info_string() -> *const ::std::os::raw::c_char;
}

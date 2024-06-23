use std::{
    char,
    io::{Cursor, Write},
    num::ParseIntError,
};

use gc_arena::Gc;
use thiserror::Error;

use crate::{
    meta_ops::{self, MetaResult},
    Context, Error, Function, Value,
};

// Could replace this with the memchr crate, if needed
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

#[derive(Debug, Error)]
enum FormatError {
    #[error("invalid format specifier {:?}", *.0 as char)]
    BadSpec(u8),
    #[error("invalid format specifier; precision is limited to {}", u8::MAX)]
    BadPrecision,
    #[error("invalid format specifier; width is limited to {}", u8::MAX)]
    BadWidth,
    #[error("invalid format specifier; flag is not supported for {}", *.spec as char)]
    BadFlag { spec: u8, invalid_flags: Flags },
    #[error("missing value for format specifier {:?}", *.0 as char)]
    MissingValue(u8),
    #[error("value of wrong type for format specifier {:?}; expected {}, found {}", *.0 as char, .1, .2)]
    BadValueType(u8, &'static str, &'static str),
    #[error("value out of range for format specifier {:?}", *.0 as char)]
    ValueOutOfRange(u8),
    #[error("weird floating point value?")]
    BadFloat,
}

/// Implementation of printf
///
/// Supported flags:
/// - `#`: alternate mode
/// - `-`: left align
/// - `0`: zero pad
/// - `+`: include sign
/// - ` `: include space in sign position for positive numbers
///
/// Unsupported flags:
/// - `'`: digit group separator
///
/// Width and precision are supported, but are limited to 255.
///
/// Unsupported conversions:
/// - `%n` (sadly, no turing complete printf)
/// - `%b`, `%B` (booleans)
/// - value length (`hh`, `h`, `l`, `ll`, `L`, etc)
///
/// Additional conversions:
/// - `%q` - print an escaped Lua literal value (escapes strings, prints floats as hex)

const FMT_SPEC: u8 = b'%';

// TODO: useful Debug impl for flags for errors?
#[derive(Debug, Copy, Clone)]
pub struct Flags(u8);
impl Flags {
    const NONE: Self = Self(0);
    const ALL: Self = Self(u8::MAX);
    const UINT: Self =
        Self(Self::LEFT_ALIGN.0 | Self::ZERO_PAD.0 | Self::WIDTH.0 | Self::PRECISION.0);
    const SINT: Self = Self(Self::UINT.0 | Self::SIGN_FORCE.0 | Self::SIGN_SPACE.0);

    const ALTERNATE: Self = Self(1 << 0);
    const LEFT_ALIGN: Self = Self(1 << 1);
    const ZERO_PAD: Self = Self(1 << 2);
    const SIGN_FORCE: Self = Self(1 << 3);
    const SIGN_SPACE: Self = Self(1 << 4);
    const WIDTH: Self = Self(1 << 5);
    const PRECISION: Self = Self(1 << 6);
}
impl Flags {
    fn has(self, flag: Flags) -> bool {
        self.0 & flag.0 == flag.0
    }
}
impl std::ops::BitOr for Flags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for Flags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

// Note: if width is specified by a argument, it will be interpreted
// as its absolute value, setting the left align flag if negative.
struct FormatSpecifier {
    spec: u8,
    flags: Flags,
    width: OptionalArg,
    precision: OptionalArg,
}
#[derive(Copy, Clone)]
enum OptionalArg {
    None,
    Arg,
    Specified(u8),
}
struct CommonFormatArgs {
    width: Option<usize>,
    precision: Option<usize>,
    left_align: bool,
    zero_pad: bool,
    alternate: bool,
    upper: bool,
    flags: Flags,
}

impl FormatSpecifier {
    fn check_flags(&self, allowed: Flags) -> Result<(), FormatError> {
        if self.flags.0 & !allowed.0 != 0 {
            Err(FormatError::BadFlag {
                spec: self.spec,
                invalid_flags: Flags(self.flags.0 & !allowed.0),
            })
        } else {
            Ok(())
        }
    }
    // Gets an argument for a format specifier (width/precision, popping
    // from the value stack if needed.
    //
    // Limited to 255, but returned as usize for convenience.
    //
    // Returns the argument value, and a boolean of whether it was negative.
    fn get_arg<'gc>(
        &self,
        arg: OptionalArg,
        values: &mut impl Iterator<Item = Value<'gc>>,
    ) -> Result<(Option<usize>, bool), FormatError> {
        match arg {
            OptionalArg::None => Ok((None, false)),
            OptionalArg::Arg => {
                let int = self.next_int(values)?;
                let negative = int < 0;
                let byte: u8 = int
                    .abs()
                    .try_into()
                    .map_err(|_| FormatError::ValueOutOfRange(self.spec))?;
                Ok((Some(byte as usize), negative))
            }
            OptionalArg::Specified(val) => Ok((Some(val as usize), false)),
        }
    }
    fn common_args<'gc>(
        &self,
        values: &mut impl Iterator<Item = Value<'gc>>,
    ) -> Result<CommonFormatArgs, FormatError> {
        let (width, width_neg) = self.get_arg(self.width, values)?;
        let (precision, _) = self.get_arg(self.precision, values)?;
        let left_align = self.flags.has(Flags::LEFT_ALIGN) || width_neg;
        let zero_pad = self.flags.has(Flags::ZERO_PAD) && !left_align;
        let alternate = self.flags.has(Flags::ALTERNATE);
        Ok(CommonFormatArgs {
            width,
            precision,
            left_align,
            zero_pad,
            alternate,
            upper: self.spec.is_ascii_uppercase(),
            flags: self.flags,
        })
    }

    fn next_value<'gc>(
        &self,
        values: &mut impl Iterator<Item = Value<'gc>>,
    ) -> Result<Value<'gc>, FormatError> {
        values
            .next()
            .ok_or_else(|| FormatError::MissingValue(self.spec))
    }
    fn next_int<'gc>(
        &self,
        values: &mut impl Iterator<Item = Value<'gc>>,
    ) -> Result<i64, FormatError> {
        let val = self.next_value(values)?;
        let int = val
            .to_integer()
            .ok_or_else(|| FormatError::BadValueType(self.spec, "integer", val.type_name()))?;
        Ok(int)
    }
    fn next_float<'gc>(
        &self,
        values: &mut impl Iterator<Item = Value<'gc>>,
    ) -> Result<f64, FormatError> {
        let val = self.next_value(values)?;
        let float = val
            .to_number()
            .ok_or_else(|| FormatError::BadValueType(self.spec, "number", val.type_name()))?;
        Ok(float)
    }
}
impl CommonFormatArgs {
    fn sign_char(&self, is_negative: bool) -> Option<u8> {
        if is_negative {
            Some(b'-')
        } else if self.flags.has(Flags::SIGN_FORCE) {
            Some(b'+')
        } else if self.flags.has(Flags::SIGN_SPACE) {
            Some(b' ')
        } else {
            None
        }
    }
}

struct PeekableIter<'a> {
    base: &'a [u8],
    cur: &'a [u8],
}
impl<'a> PeekableIter<'a> {
    fn new(s: &'a [u8]) -> Self {
        Self { base: s, cur: s }
    }
    fn peek(&mut self) -> Option<u8> {
        self.cur.get(0).copied()
    }
    fn next(&mut self) -> Option<u8> {
        let v = self.cur.get(0).copied();
        self.cur = &self.cur[1..];
        v
    }
    fn cur_index(&self) -> usize {
        self.cur.as_ptr() as usize - self.base.as_ptr() as usize
    }
}

fn try_parse_optional_arg(iter: &mut PeekableIter<'_>) -> Result<OptionalArg, ParseIntError> {
    match iter.peek() {
        Some(b'*') => {
            iter.next();
            Ok(OptionalArg::Arg)
        }
        Some(b'0'..=b'9') => {
            let rest = &iter.cur[1..];
            let len = 1 + rest
                .iter()
                .position(|c| !matches!(c, b'0'..=b'9'))
                .unwrap_or(rest.len());

            // Safety: We just verified that the string is only composed
            // of ASCII characters between 0 and 9.
            let slice = unsafe { std::str::from_utf8_unchecked(&iter.cur[..len]) };

            let num = slice.parse::<u8>()?;

            iter.cur = &iter.cur[len..];
            Ok(OptionalArg::Specified(num))
        }
        _ => Ok(OptionalArg::None),
    }
}

fn write_padding<W>(w: &mut W, byte: u8, count: usize) -> Result<(), std::io::Error>
where
    W: Write,
{
    // TODO: check efficiency
    let buf = [byte; 16];
    let mut remaining = count;
    while remaining > 0 {
        match w.write(&buf[..remaining.min(buf.len())]) {
            Ok(n) => remaining -= n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn integer_length(i: u64) -> usize {
    1 + i.checked_ilog(10).unwrap_or(0) as usize
}
fn integer_length_hex(i: u64) -> usize {
    1 + i.checked_ilog2().unwrap_or(0) as usize / 4
}
fn integer_length_octal(i: u64) -> usize {
    1 + i.checked_ilog2().unwrap_or(0) as usize / 3
}
fn integer_length_binary(i: u64) -> usize {
    1 + i.checked_ilog2().unwrap_or(0) as usize
}

fn format_into_buffer<'a>(
    buf: &'a mut [u8],
    args: std::fmt::Arguments<'_>,
) -> Result<&'a str, std::io::Error> {
    let mut buf = Cursor::new(buf);
    write!(&mut buf, "{}", args)?;
    let len = buf.position() as usize;
    let slice = &buf.into_inner()[..len];

    // Safety: write! can only output valid utf8.
    let str = unsafe { std::str::from_utf8_unchecked(slice) };
    Ok(str)
}

fn pad_num_before<W: Write>(
    w: &mut W,
    len: usize,
    width: usize,
    zero_width: usize,
    left_align: bool,
    prefix: &[u8],
) -> Result<PadScope, std::io::Error> {
    // right: [    ][-][0000][nnnn]
    // left:  [-][0000][nnnn][    ]
    let zero_padding = zero_width.saturating_sub(len);
    let space_padding = width.saturating_sub(zero_padding + prefix.len() + len);
    if space_padding > 0 && !left_align {
        write_padding(w, b' ', space_padding)?;
    }
    if !prefix.is_empty() {
        w.write_all(prefix)?;
    }
    if zero_padding > 0 {
        write_padding(w, b'0', zero_padding)?;
    }
    Ok(PadScope {
        trailing_padding: if left_align { space_padding } else { 0 },
    })
}
#[must_use]
struct PadScope {
    trailing_padding: usize,
}
impl PadScope {
    fn finish_pad<W: Write>(self, w: &mut W) -> Result<(), std::io::Error> {
        if self.trailing_padding > 0 {
            write_padding(w, b' ', self.trailing_padding)?;
        }
        Ok(())
    }
}

fn strip_nonsignificant_zeroes(str: &str) -> &str {
    if let Some(last_nonzero) = str.bytes().rposition(|p| p != b'0') {
        if let Some(decimal) = str[..last_nonzero + 1].rfind('.') {
            if decimal == last_nonzero {
                return &str[..last_nonzero];
            } else {
                return &str[..last_nonzero + 1];
            }
        }
    }
    str
}

fn write_hex_float<W: Write>(
    w: &mut W,
    float: f64,
    args: CommonFormatArgs,
) -> Result<(), std::io::Error> {
    let sign = args.sign_char(float.is_sign_negative());
    let preserve_decimal = args.alternate;

    const F64_EXPONENT_BITS: u32 = 11;
    const F64_MANTISSA_BITS: u32 = 52;

    let width = args.width.unwrap_or(0);
    let precision = args
        .precision
        .unwrap_or(F64_MANTISSA_BITS.div_ceil(4) as usize);

    if !float.is_finite() {
        let s = match (float.is_infinite(), args.upper) {
            (true, false) => "inf",
            (true, true) => "INF",
            (false, false) => "nan",
            (false, true) => "NAN",
        };
        let pad = pad_num_before(w, s.len(), width, 0, args.left_align, sign.as_slice())?;
        write!(w, "{s}")?;
        pad.finish_pad(w)?;
    } else {
        // TODO: test subnormals
        let mut head = !(f64::is_subnormal(float) || float == 0.0) as usize;

        let bits = f64::to_bits(float);
        let exp_bits = (bits >> F64_MANTISSA_BITS) & ((1 << F64_EXPONENT_BITS) - 1);
        let mut exp = exp_bits as i16 - ((1 << (F64_EXPONENT_BITS - 1)) - 1);
        let mantissa = bits & ((1 << F64_MANTISSA_BITS) - 1);

        if float == 0.0 {
            exp = 0;
        }

        let used_mantissa_bits = precision * 4;
        let mantissa = match used_mantissa_bits as u32 {
            0 => {
                let base = 1;
                let remaining_bits = F64_MANTISSA_BITS;
                let remaining = mantissa & ((1 << remaining_bits) - 1);
                head = if remaining.leading_zeros() == 64 - remaining_bits {
                    if (remaining & !(1 << (remaining_bits - 1))).leading_zeros() == 64 {
                        // Round to even
                        (base + 1) & !1
                    } else {
                        base + 1
                    }
                } else {
                    base
                };
                0
            }
            F64_MANTISSA_BITS => mantissa,
            1..=F64_MANTISSA_BITS => {
                let base = mantissa >> (F64_MANTISSA_BITS - used_mantissa_bits as u32);
                let remaining_bits = F64_MANTISSA_BITS - used_mantissa_bits as u32;
                let remaining = mantissa & ((1 << remaining_bits) - 1);
                // println!("{:0w$b} ({}, {})", remaining, remaining.leading_zeros(), remaining_bits, w = remaining_bits as usize);
                if remaining.leading_zeros() == 64 - remaining_bits {
                    if (remaining & !(1 << (remaining_bits - 1))).leading_zeros() == 64 {
                        // Round to even
                        (base + 1) & !1
                    } else {
                        base + 1
                    }
                } else {
                    base
                }
            }
            _ => mantissa,
        };

        let prefix: &[u8] = match (sign, args.upper) {
            (None, false) => b"0x",
            (Some(b'-'), false) => b"-0x",
            (Some(b'+'), false) => b"+0x",
            (Some(b' '), false) => b" 0x",
            (None, true) => b"0X",
            (Some(b'-'), true) => b"-0X",
            (Some(b'+'), true) => b"+0X",
            (Some(b' '), true) => b" 0X",
            _ => unreachable!(),
        };
        let zero_width = if args.zero_pad {
            width.saturating_sub(prefix.len())
        } else {
            0
        };

        if precision != 0 {
            let m_width = precision;
            let len = 2 + m_width + 1 + 1 + integer_length(exp.unsigned_abs() as u64);

            let pad = pad_num_before(w, len, width, zero_width, args.left_align, prefix)?;
            if args.upper {
                write!(w, "{head}.{mantissa:0m_width$X}P{exp:+}")?;
            } else {
                write!(w, "{head}.{mantissa:0m_width$x}p{exp:+}")?;
            }
            pad.finish_pad(w)?;
        } else {
            let len = 3 + preserve_decimal as usize + integer_length(exp.unsigned_abs() as u64);

            let p = if args.upper { 'P' } else { 'p' };
            let pad = pad_num_before(w, len, width, zero_width, args.left_align, prefix)?;
            if preserve_decimal {
                write!(w, "{head}.{p}{exp:+}")?;
            } else {
                write!(w, "{head}{p}{exp:+}")?;
            }
            pad.finish_pad(w)?;
        }
    }
    Ok(())
}

// TODO: this could be made interruptable

// Interesting optimization idea: cache an intermediate bytecode
// when run on interned strings; could give length estimates for
// buffer sizing.
pub fn string_format<'gc, W>(
    ctx: Context<'gc>,
    w: &mut W,
    str: &[u8],
    mut values: impl Iterator<Item = Value<'gc>>,
) -> Result<usize, Error<'gc>>
where
    W: Write,
{
    let mut float_buf = [0u8; 300];

    let mut index = 0;

    while let Some(next) = memchr(FMT_SPEC, &str[index..]).map(|n| n + index) {
        if next != index {
            w.write_all(&str[index..next])?;
        }

        let mut iter = PeekableIter::new(&str[next + 1..]);

        let mut flags = Flags::NONE;
        #[rustfmt::skip]
        let _ = loop {
            match iter.peek() {
                Some(b'#') => { iter.next(); flags |= Flags::ALTERNATE; },
                Some(b'-') => { iter.next(); flags |= Flags::LEFT_ALIGN; },
                Some(b'+') => { iter.next(); flags |= Flags::SIGN_FORCE; },
                Some(b' ') => { iter.next(); flags |= Flags::SIGN_SPACE; },
                Some(b'0') => { iter.next(); flags |= Flags::ZERO_PAD; },
                _ => break,
            }
        };
        let width = try_parse_optional_arg(&mut iter).map_err(|_| FormatError::BadWidth)?;
        if !matches!(width, OptionalArg::None) {
            flags |= Flags::WIDTH;
        }

        let precision;
        if let Some(b'.') = iter.peek() {
            iter.next();
            precision =
                match try_parse_optional_arg(&mut iter).map_err(|_| FormatError::BadPrecision)? {
                    // Weirdly, %.f is a fine format specifier, and is treated as %.0f
                    OptionalArg::None => OptionalArg::Specified(0),
                    arg => arg,
                };
            flags |= Flags::PRECISION;
        } else {
            precision = OptionalArg::None;
        }

        // We do not support length modifiers (ie. l, ll, etc)

        let spec = iter.next().ok_or_else(|| FormatError::BadSpec(FMT_SPEC))?;
        let spec_end = next + 1 + iter.cur_index();

        // println!(
        //     "processed spec {:?}",
        //     std::string::String::from_utf8_lossy(&str[next..spec_end])
        // );
        index = spec_end;

        let spec = FormatSpecifier {
            spec,
            flags,
            width,
            precision,
        };

        match spec.spec {
            b'%' => {
                spec.check_flags(Flags::NONE)?;
                w.write_all(b"%")?;
            }
            b'c' => {
                // char
                spec.check_flags(Flags::LEFT_ALIGN | Flags::WIDTH)?;

                let (width, width_neg) = spec.get_arg(spec.width, &mut values)?;
                let left_align = spec.flags.has(Flags::LEFT_ALIGN) || width_neg;

                let int = spec.next_int(&mut values)?;
                let byte: u8 = (int % 256) as u8;

                let width = width.unwrap_or(1);
                if width > 1 && !left_align {
                    write_padding(w, b' ', width - 1)?;
                }

                w.write_all(&[byte])?;

                if width > 1 && left_align {
                    write_padding(w, b' ', width - 1)?;
                }
            }
            b's' => {
                // string
                spec.check_flags(Flags::LEFT_ALIGN | Flags::WIDTH | Flags::PRECISION)?;

                let (width, width_neg) = spec.get_arg(spec.width, &mut values)?;
                let (precision, _) = spec.get_arg(spec.precision, &mut values)?;
                let left_align = spec.flags.has(Flags::LEFT_ALIGN) || width_neg;

                let val = spec.next_value(&mut values)?;
                let val = match meta_ops::tostring(ctx, val)? {
                    MetaResult::Value(val) => val,
                    MetaResult::Call(_) => {
                        // this makes the entire thing a state machine...
                        todo!("support tostring calls")
                    }
                };
                let string = val.into_string(ctx).ok_or_else(|| {
                    FormatError::BadValueType(spec.spec, "string", val.type_name())
                })?;

                // our strings are not necessarily unicode, so we can't call out to
                // the stdlib impl...

                let len = string.len() as usize;

                let precision = precision.unwrap_or(len).min(len);
                let width = width.unwrap_or(precision);

                let padding = width.saturating_sub(precision);
                if padding > 0 && !left_align {
                    write_padding(w, b' ', padding)?;
                }

                w.write_all(&string[..precision])?;

                if padding > 0 && left_align {
                    write_padding(w, b' ', padding)?;
                }
            }
            b'd' | b'i' => {
                // signed int
                spec.check_flags(Flags::SINT)?;
                let args = spec.common_args(&mut values)?;

                let int = spec.next_int(&mut values)?;
                let len = integer_length(int.unsigned_abs());
                let sign = args.sign_char(int < 0);

                let width = args.width.unwrap_or(0);
                let default_precision = if args.zero_pad {
                    width - sign.is_some() as usize
                } else {
                    0
                };
                let precision = args.precision.unwrap_or(default_precision);

                let pad =
                    pad_num_before(w, len, width, precision, args.left_align, sign.as_slice())?;
                write!(w, "{}", int.unsigned_abs())?;
                pad.finish_pad(w)?;
            }
            b'u' => {
                // unsigned int
                spec.check_flags(Flags::UINT)?;
                let args = spec.common_args(&mut values)?;

                let int = spec.next_int(&mut values)? as u64;

                let len = integer_length(int);

                let width = args.width.unwrap_or(0);
                let default_precision = if args.zero_pad { width } else { 0 };
                let precision = args.precision.unwrap_or(default_precision);

                let pad =
                    pad_num_before(w, len, width, precision, args.left_align, None.as_slice())?;
                write!(w, "{}", int)?;
                pad.finish_pad(w)?;
            }
            b'o' => {
                // octal unsigned int
                spec.check_flags(Flags::UINT | Flags::ALTERNATE)?;
                let args = spec.common_args(&mut values)?;

                let prefix: &[u8] = match spec.flags.has(Flags::ALTERNATE) {
                    true => b"0",
                    false => b"",
                };

                let int = spec.next_int(&mut values)? as u64;

                let len = integer_length_octal(int);

                let width = args.width.unwrap_or(0);
                let default_precision = if args.zero_pad {
                    width.saturating_sub(prefix.len())
                } else {
                    0
                };
                let precision = args.precision.unwrap_or(default_precision);

                let pad = pad_num_before(w, len, width, precision, args.left_align, prefix)?;
                write!(w, "{:o}", int as u64)?;
                pad.finish_pad(w)?;
            }
            b'x' | b'X' => {
                // hex unsigned int
                spec.check_flags(Flags::UINT | Flags::ALTERNATE)?;
                let args = spec.common_args(&mut values)?;

                let prefix: &[u8] = match (spec.flags.has(Flags::ALTERNATE), args.upper) {
                    (true, false) => b"0x",
                    (true, true) => b"0X",
                    (false, _) => b"",
                };

                let int = spec.next_int(&mut values)? as u64;

                let len = integer_length_hex(int);

                let width = args.width.unwrap_or(0);
                let default_precision = if args.zero_pad {
                    width.saturating_sub(prefix.len())
                } else {
                    0
                };
                let precision = args.precision.unwrap_or(default_precision);

                let pad = pad_num_before(w, len, width, precision, args.left_align, prefix)?;
                if args.upper {
                    write!(w, "{:X}", int as u64)?;
                } else {
                    write!(w, "{:x}", int as u64)?;
                }
                pad.finish_pad(w)?;
            }
            b'b' | b'B' => {
                // binary unsigned int
                spec.check_flags(Flags::UINT | Flags::ALTERNATE)?;
                let args = spec.common_args(&mut values)?;

                let prefix: &[u8] = match (spec.flags.has(Flags::ALTERNATE), args.upper) {
                    (true, false) => b"0b",
                    (true, true) => b"0B",
                    (false, _) => b"",
                };

                let int = spec.next_int(&mut values)? as u64;

                let len = integer_length_binary(int);

                let width = args.width.unwrap_or(0);
                let default_precision = if args.zero_pad {
                    width.saturating_sub(prefix.len())
                } else {
                    0
                };
                let precision = args.precision.unwrap_or(default_precision);

                let pad = pad_num_before(w, len, width, precision, args.left_align, prefix)?;
                write!(w, "{:b}", int as u64)?;
                pad.finish_pad(w)?;
            }
            c @ (b'g' | b'G' | b'e' | b'E' | b'f' | b'F') => {
                let is_compact = matches!(c, b'g' | b'G');
                let is_exp = matches!(c, b'e' | b'E' | b'g' | b'G');

                spec.check_flags(Flags::ALL)?;
                let args = spec.common_args(&mut values)?;
                let preserve_decimal = args.alternate;

                let float = spec.next_float(&mut values)?;
                let sign = args.sign_char(float.is_sign_negative());

                let width = args.width.unwrap_or(0);
                let precision = args.precision.unwrap_or(6);

                if !float.is_finite() {
                    let s = match (float.is_infinite(), args.upper) {
                        (true, false) => "inf",
                        (true, true) => "INF",
                        (false, false) => "nan",
                        (false, true) => "NAN",
                    };
                    let pad =
                        pad_num_before(w, s.len(), width, 0, args.left_align, sign.as_slice())?;
                    write!(w, "{s}")?;
                    pad.finish_pad(w)?;
                } else if is_exp {
                    let p = if is_compact {
                        precision.saturating_sub(1)
                    } else {
                        precision
                    };
                    let str = format_into_buffer(&mut float_buf, format_args!("{:+.p$e}", float))?;

                    let idx = str.rfind('e').ok_or(FormatError::BadFloat)?;
                    let exp = str[idx + 1..]
                        .parse::<i16>()
                        .map_err(|_| FormatError::BadFloat)?;
                    let exp_len = str[idx + 1..].len();

                    // Implementation of %g, following the description of the algorithm
                    // in Python's documentation:
                    // https://docs.python.org/3/library/string.html#format-specification-mini-language
                    if is_compact && exp >= -4 && (exp as i64) < precision as i64 {
                        let p = (precision as i64 - 1 - exp as i64) as usize;

                        let str;
                        if preserve_decimal {
                            // Add a decimal at the end, in case Rust doesn't generate one; then strip it out
                            let s = format_into_buffer(
                                &mut float_buf,
                                format_args!("{:+.p$}.", float),
                            )?;
                            if s[1..s.len() - 1].contains('.') {
                                str = &s[1..s.len() - 1];
                            } else {
                                str = &s[1..];
                            }
                        } else {
                            let s =
                                format_into_buffer(&mut float_buf, format_args!("{:+.p$}", float))?;
                            str = strip_nonsignificant_zeroes(&s[1..]);
                        }

                        let len = str.len();
                        let zero_width = if args.zero_pad { width } else { 0 };

                        let pad = pad_num_before(
                            w,
                            len,
                            width,
                            zero_width,
                            args.left_align,
                            sign.as_slice(),
                        )?;
                        write!(w, "{}", str)?;
                        pad.finish_pad(w)?;
                    } else {
                        // [   ][-][000][a.bbb][e][+EE]
                        let mut mantissa = &str[1..idx];
                        if is_compact && !preserve_decimal {
                            mantissa = strip_nonsignificant_zeroes(mantissa);
                        }
                        let e = if args.upper { 'E' } else { 'e' };

                        let exp_len = 1 + exp_len.max(2);
                        let len = mantissa.len() + 1 + exp_len;
                        let zero_width = if args.zero_pad { width } else { 0 };

                        let fallback_dec = preserve_decimal && !str.contains('.');

                        if !fallback_dec {
                            let pad = pad_num_before(
                                w,
                                len,
                                width,
                                zero_width,
                                args.left_align,
                                sign.as_slice(),
                            )?;
                            write!(w, "{mantissa}{e}{exp:+03}")?;
                            pad.finish_pad(w)?;
                        } else {
                            let pad = pad_num_before(
                                w,
                                len + 1,
                                width,
                                zero_width,
                                args.left_align,
                                sign.as_slice(),
                            )?;
                            write!(w, "{mantissa}.{e}{exp:+03}")?;
                            pad.finish_pad(w)?;
                        }
                    }
                } else {
                    // normal float
                    // This can be larger than any reasonable buffer, so we have
                    // to forward everything to std

                    // TODO: cannot support the '#' preserving decimal mode
                    // string.format("'%#.0f'", 1) should result in "1."
                    match (args.left_align, args.zero_pad, sign) {
                        (false, false, None | Some(b'-')) => {
                            write!(w, "{float:width$.precision$}")?
                        }
                        (false, true, None | Some(b'-')) => {
                            write!(w, "{float:>0width$.precision$}")?
                        }
                        (false, false, Some(b'+')) => write!(w, "{float:+width$.precision$}")?,
                        (false, true, Some(b'+')) => write!(w, "{float:>+0width$.precision$}")?,
                        (false, false, Some(b' ')) => write!(w, " {float:width$.precision$}")?,
                        (false, true, Some(b' ')) => write!(w, " {float:>0width$.precision$}")?,
                        (true, _, None | Some(b'-')) => write!(w, "{float:<width$.precision$}")?,
                        (true, _, Some(b'+')) => write!(w, "{float:<+width$.precision$}")?,
                        (true, _, Some(b' ')) => write!(w, " {float:<width$.precision$}")?,
                        _ => unreachable!(),
                    }
                }
            }
            b'a' | b'A' => {
                // float, hex
                spec.check_flags(Flags::ALL)?;
                let common_args = spec.common_args(&mut values)?;

                let float = spec.next_float(&mut values)?;
                write_hex_float(w, float, common_args)?;
            }
            b'p' => {
                // pointer
                spec.check_flags(Flags::LEFT_ALIGN | Flags::WIDTH)?;

                let (width, width_neg) = spec.get_arg(spec.width, &mut values)?;
                let left_align = spec.flags.has(Flags::LEFT_ALIGN) || width_neg;

                // TODO: is an intentional address-leak a bad idea?  Defeats ASLR
                // (though addrs are currently already exposed through tostring on fns/tables)
                let val = spec.next_value(&mut values)?;
                let ptr = match val {
                    Value::Nil => 0,
                    Value::Boolean(_) => 0,
                    Value::Integer(_) => 0,
                    Value::Number(_) => 0,
                    Value::String(str) => str.as_ptr() as usize,
                    Value::Table(t) => Gc::as_ptr(t.into_inner()) as usize,
                    Value::Function(Function::Closure(c)) => Gc::as_ptr(c.into_inner()) as usize,
                    Value::Function(Function::Callback(c)) => Gc::as_ptr(c.into_inner()) as usize,
                    Value::Thread(t) => Gc::as_ptr(t.into_inner()) as usize,
                    Value::UserData(u) => Gc::as_ptr(u.into_inner()) as usize,
                };

                let prefix: &[u8] = b"0x";

                let len = integer_length_hex(ptr as u64);

                let width = width.unwrap_or(0);
                let pad = pad_num_before(w, len, width, 0, left_align, prefix)?;
                write!(w, "{:x}", ptr)?;
                pad.finish_pad(w)?;
            }
            b'q' => {
                // Lua escape
                spec.check_flags(Flags::NONE)?;

                let val = spec.next_value(&mut values)?;
                match val {
                    Value::Nil => write!(w, "nil")?,
                    Value::Boolean(b) => write!(w, "{}", b)?,
                    Value::Integer(i) => write!(w, "{}", i)?,
                    Value::Number(n) => {
                        write_hex_float(
                            w,
                            n,
                            CommonFormatArgs {
                                width: None,
                                precision: None,
                                left_align: false,
                                zero_pad: false,
                                alternate: false,
                                upper: false,
                                flags,
                            },
                        )?;
                    }
                    Value::String(str) => {
                        // TODO: check string escaping
                        write!(w, "\"")?;
                        for c in str.as_bytes() {
                            // TODO: handling of newlines?
                            match *c {
                                c @ (b'\t' | b' ' | b'!' | b'#'..=b'[' | b']'..=b'~') => {
                                    write!(w, "{}", c as char)?;
                                }
                                c @ (b'\\' | b'"') => write!(w, "\\{}", c as char)?,
                                b'\n' => write!(w, "\\n")?,
                                b'\r' => write!(w, "\\r")?,
                                c => write!(w, "\\x{:02}", c)?,
                            }
                        }
                        write!(w, "\"")?;
                    }
                    _ => {
                        return Err(FormatError::BadValueType(
                            spec.spec,
                            "constant",
                            val.type_name(),
                        )
                        .into());
                    }
                }
            }
            b'n' => {
                // printf oriented programming, as a treat? :)
            }
            c => return Err(FormatError::BadSpec(c).into()),
        }

        // Must make forward progress
        assert!(index > next);
    }

    if index < str.len() {
        w.write_all(&str[index..])?;
    }

    Ok(0)
}

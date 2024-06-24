
local function dprint(...)
    print(string.format("%q", ...))
end

local function assert_eq(val, exp)
    -- dprint(val)
    if val ~= exp then
        error(string.format("assertion failed; expected %q but found %q", exp, val))
    end
end


assert_eq(string.format("abc%sdef", "iii"), "abciiidef")

assert_eq(string.format("%s__", 123), "123__")
assert_eq(string.format("__%s", 321), "__321")

do
    assert_eq(string.format("%d %d %d", 1, 2, 3), "1 2 3")
end


do -- string width, truncating
    assert_eq(string.format("%s", "example"), "example")
    assert_eq(string.format("%5s", "a"), "    a")
    assert_eq(string.format("%-5s", "a"), "a    ")
    assert_eq(string.format("%-s", "example"), "example")

    -- assert_eq(string.format("%*s", 3, "a"), "  a")
    -- assert_eq(string.format("%-*s", 3, "a"), "a  ")
    -- assert_eq(string.format("%*s", -3, "a"), "a  ")

    assert_eq(string.format("%.3s", "example"), "exa")
    assert_eq(string.format("%.s", "example"), "")
    assert_eq(string.format("%5.2s", "example"), "   ex")
    assert_eq(string.format("%-5.2s", "example"), "ex   ")
    assert_eq(string.format("%2.s", "example"), "  ")

    assert_eq(string.format("%.10s", "example"), "example")
    assert_eq(string.format("%10.10s", "example"), "   example")
    assert_eq(string.format("%10.3s", "example"), "       exa")

    assert_eq(string.format("%4s", ""), "    ")
    assert_eq(string.format("%4.2s", ""), "    ")
    assert_eq(string.format("%4.10s", ""), "    ")

    assert_eq(string.format("'%8d'", -3), "'      -3'")
    assert_eq(string.format("'%8.4d'", 3), "'    0003'")
    assert_eq(string.format("'%8.4d'", -3), "'   -0003'")
    assert_eq(string.format("'%+8.4d'", 3), "'   +0003'")
    assert_eq(string.format("'% 8.4d'", 3), "'    0003'")
    assert_eq(string.format("'%08.4d'", 3), "'    0003'")
    assert_eq(string.format("'%+08.4d'", 3), "'   +0003'")
    assert_eq(string.format("'%-08.4d'", 3), "'0003    '")
    assert_eq(string.format("'%-+08.4d'", 3), "'+0003   '")
    assert_eq(string.format("'%+08.4d'", 123456789), "'+123456789'")

    assert_eq(string.format("'% 1.4d'", 1234), "' 1234'")
    assert_eq(string.format("'% 5.4d'", 1234), "' 1234'")

    -- '+' takes precedence over ' '
    assert_eq(string.format("'% +d'", 1), "'+1'")
    assert_eq(string.format("'%+ d'", 1), "'+1'")
    assert_eq(string.format("'%+ d'", -1), "'-1'")

    assert_eq(string.format("'%+08d'", 3), "'+0000003'")
    assert_eq(string.format("'% 08d'", 3), "' 0000003'")
    assert_eq(string.format("'%08d'", 3), "'00000003'")
    assert_eq(string.format("'%+08d'", -3), "'-0000003'")
    assert_eq(string.format("'% 08d'", -3), "'-0000003'")
    assert_eq(string.format("'%08d'", -3), "'-0000003'")

    -- dprint("unsigned")

    assert_eq(string.format("'%8.4u'", 3), "'    0003'")
    assert_eq(string.format("'%8.4u'", -3), "'18446744073709551613'")
    assert_eq(string.format("'%08.4u'", 3), "'    0003'")
    assert_eq(string.format("'%-08.4u'", 3), "'0003    '")

    assert_eq(string.format("'%1.4u'", 1234), "'1234'")
    assert_eq(string.format("'%5.4u'", 1234), "' 1234'")

    assert_eq(string.format("'%08u'", 3), "'00000003'")
    assert_eq(string.format("'%08u'", -3), "'18446744073709551613'")
    assert_eq(string.format("'%8u'", 3), "'       3'")

    assert_eq(string.format("'%-8u'", 3), "'3       '")
    assert_eq(string.format("'%-08u'", 3), "'3       '")

    assert_eq(string.format("%08d",  1), "00000001")
    assert_eq(string.format("%08d", -1), "-0000001")
    assert_eq(string.format("%.8d",  1), "00000001")
    assert_eq(string.format("%.8d", -1), "-00000001")

    assert_eq(string.format("%+08d",  1), "+0000001")
    assert_eq(string.format("%+08d", -1), "-0000001")
    assert_eq(string.format("%+.8d",  1), "+00000001")
    assert_eq(string.format("%+.8d", -1), "-00000001")

    assert_eq(string.format("%#16.8x", 235678), "      0x0003989e")


    assert_eq(string.format("%g", 0), "0")
    assert_eq(string.format("%g", -0.0), "-0")
    assert_eq(string.format("%+g", 0), "+0")
    assert_eq(string.format("% g", 0), " 0")

    assert_eq(string.format("%g", 1), "1")
    assert_eq(string.format("%g", -1), "-1")
    assert_eq(string.format("%+g", 1), "+1")
    assert_eq(string.format("% g", 1), " 1")

    assert_eq(string.format("%g", 1.500001), "1.5")
    assert_eq(string.format("%g", 1.50001), "1.50001")
    assert_eq(string.format("%.1g", 1.5), "2")
    assert_eq(string.format("%g", 1000), "1000")
    assert_eq(string.format("%g", 100000), "100000")
    assert_eq(string.format("%g", 1000000), "1e+06")

    assert_eq(string.format("%8g", 1), "       1")
    assert_eq(string.format("%8g", 1.500001), "     1.5")
    assert_eq(string.format("%8g", 1.50001), " 1.50001")
    assert_eq(string.format("%8.1g", 1.5), "       2")
    assert_eq(string.format("%8g", 1000), "    1000")
    assert_eq(string.format("%8g", 100000), "  100000")
    assert_eq(string.format("%8g", 1000000), "   1e+06")
    assert_eq(string.format("%8G", 1000000), "   1E+06")
    assert_eq(string.format("%8e", 1000000), "1.000000e+06")
    assert_eq(string.format("%8E", 1000000), "1.000000E+06")

    -- dprint(string.format("%08g", 1))
    -- dprint(string.format("%08g", 1.500001))
    -- dprint(string.format("%08g", 1.50001))
    -- dprint(string.format("%08.1g", 1.5))
    -- dprint(string.format("%08g", 1000))
    -- dprint(string.format("%08g", 100000))
    -- dprint(string.format("%08g", 1000000))

    -- dprint(string.format("%-08g", 1))
    -- dprint(string.format("%-08g", 1.500001))
    -- dprint(string.format("%-08g", 1.50001))
    -- dprint(string.format("%-08.1g", 1.5))
    -- dprint(string.format("%-08g", 1000))
    -- dprint(string.format("%-08g", 100000))
    -- dprint(string.format("%-08g", 1000000))

    assert_eq(string.format("%g", 0/0), "-nan")
    assert_eq(string.format("%g", math.abs(0/0)), "nan")
    assert_eq(string.format("%g", 1/0), "inf")
    assert_eq(string.format("%g", -1/0), "-inf")
    assert_eq(string.format("%G", 0/0), "-NAN")
    assert_eq(string.format("%G", math.abs(0/0)), "NAN")
    assert_eq(string.format("%G", 1/0), "INF")
    assert_eq(string.format("%G", -1/0), "-INF")

    assert_eq(string.format("%05g", 0/0), " -nan")
    assert_eq(string.format("%05g", math.abs(0/0)), "  nan")
    assert_eq(string.format("%05g", 1/0), "  inf")
    assert_eq(string.format("%05g", -1/0), " -inf")

    assert_eq(string.format("%05.8g", 0/0), " -nan")
    assert_eq(string.format("%05.8g", math.abs(0/0)), "  nan")
    assert_eq(string.format("%05.8g", 1/0), "  inf")
    assert_eq(string.format("%05.8g", -1/0), " -inf")

    assert_eq(#string.format("%099.99f", 1.7976931348623158e308), 409)

    assert_eq(string.format("%0.0f", 15.1234), "15")
    assert_eq(string.format("%4.0f", 15.1234), "  15")
    assert_eq(string.format("'%-8.3f'", 15.1234), "'15.123  '")
    assert_eq(string.format("'%-+8.3f'", 15.1234), "'+15.123 '")
    assert_eq(string.format("'%0+8.3f'", 15.1234), "'+015.123'")

    -- assert_eq(string.format("'%#.0f'", 1), "'1.'") -- Can't fix this, relying on rust's std
    assert_eq(string.format("'%.0f'", 1), "'1'")
    assert_eq(string.format("'%g'", 1), "'1'")
    assert_eq(string.format("'%#g'", 1), "'1.00000'")
    assert_eq(string.format("'%#g'", 100000), "'100000.'")

    assert_eq(string.format("'%.0e'", 1), "'1e+00'")
    assert_eq(string.format("'%#.0e'", 1), "'1.e+00'")

    -- turns out PRLua doesn't support %F, even though it has different output (for nan/inf)
    assert_eq(string.format("'%f'", 1.0/0.0), "'inf'")
    assert_eq(string.format("'%e' '%E'", 1.0/0.0, 1.0/0.0), "'inf' 'INF'")
    assert_eq(string.format("'%g' '%G'", 1.0/0.0, 1.0/0.0), "'inf' 'INF'")

    assert_eq(string.format("'%8.4x'", 0xAB), "'    00ab'")
    assert_eq(string.format("'%#8.4x'", 0xAB), "'  0x00ab'")
    assert_eq(string.format("'%08.4x'", 0xAB), "'    00ab'")
    assert_eq(string.format("'%#08.4x'", 0xAB), "'  0x00ab'")
    assert_eq(string.format("'%08x'", 0xAB), "'000000ab'")
    assert_eq(string.format("'%#08x'", 0xAB), "'0x0000ab'")
    assert_eq(string.format("'%08.x'", 0xAB), "'      ab'")
    assert_eq(string.format("'%#08.x'", 0xAB), "'    0xab'")

    -- From glibc's manual:
    -- https://www.gnu.org/software/libc/manual/html_node/Floating_002dPoint-Conversions.html
    expected = {
        "|  0x0.0000p+0|       0.0000|   0.0000e+00|            0|",
        "|  0x1.0000p-1|       0.5000|   5.0000e-01|          0.5|",
        "|  0x1.0000p+0|       1.0000|   1.0000e+00|            1|",
        "| -0x1.0000p+0|      -1.0000|  -1.0000e+00|           -1|",
        "|  0x1.9000p+6|     100.0000|   1.0000e+02|          100|",
        "|  0x1.f400p+9|    1000.0000|   1.0000e+03|         1000|",
        "| 0x1.3880p+13|   10000.0000|   1.0000e+04|        1e+04|",
        "| 0x1.81c8p+13|   12345.0000|   1.2345e+04|    1.234e+04|",
        "| 0x1.86a0p+16|  100000.0000|   1.0000e+05|        1e+05|",
        "| 0x1.e240p+16|  123456.0000|   1.2346e+05|    1.235e+05|",
    }
    for i, v in ipairs({ 0, 0.5, 1, -1, 100, 1000, 10000, 12345, 1e5, 123456 }) do
        assert_eq(string.format("|%13.4a|%13.4f|%13.4e|%13.4g|", v, v, v, v), expected[i])
    end

    assert_eq(string.format("%13.4a", 0.0/0.0), "         -nan")
    assert_eq(string.format("%13.4a", 1.0/0.0), "          inf")
    assert_eq(string.format("%13.4a", math.pi), "  0x1.9220p+1")
    assert_eq(string.format("%13.4a", math.huge), "          inf")

    assert_eq(string.format("%13.4a", 0x1.00008p+0), "  0x1.0000p+0")
    assert_eq(string.format("%13.4a", 0x1.000081p+0), "  0x1.0001p+0")
    assert_eq(string.format("%13.4a", 0x1.00007fp+0), "  0x1.0000p+0")

    assert_eq(string.format("%13a", math.pi), "0x1.921fb54442d18p+1")
    assert_eq(string.format("%013.1a", math.pi), "0x000001.9p+1")
    assert_eq(string.format("%013.0a", math.pi), "0x00000002p+1")

    assert_eq(string.format("%#013.0a", 4.0), "0x0000001.p+2")
    assert_eq(string.format("%#013.0a", math.pi), "0x0000002.p+1")
end

do

    assert_eq(string.format("%q", "asdf\nabc\"def"), "\"asdf\\nabc\\\"def\"")

    assert_eq(string.format("%q", 15.345678), "0x1.eb0fcb4f1e4b4p+3")

    stringable = setmetatable({}, { __tostring = function ()
        return "abc"
    end })

    assert_eq(string.format("'%s'", stringable), "'abc'")

end

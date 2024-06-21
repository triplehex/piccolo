function is_err(f)
    return pcall(f) == false
end

do
    assert(tonumber("3.4e0") == 3.4)
    assert(tonumber("-3.4") == -3.4)
    assert(tonumber("3.4e1") == 34)
    assert(tonumber("3e.1") == nil)
    assert(tonumber("3.") == 3)
    assert(tonumber("   .4") == 0.4)
    assert(tonumber("3e3") == 3000)
    assert(tonumber("3..") == nil)
    assert(tonumber("3ee") == nil)
    assert(tonumber("3f") == nil)
    assert(tonumber("3.4", 10) == nil)
    assert(tonumber("a2", 16) == 162)
    assert(tonumber("A2", 16) == 162)
    assert(tonumber("Z0", 36) == 36 * 35)
    assert(tonumber("-ff", 16) == -255)
    assert(tonumber("") == nil)
    assert(tonumber("-") == nil)
    assert(tonumber("1 1") == nil)
    assert(tonumber({}) == nil)
    assert(is_err(function() tonumber(3, 4) end))
    assert(is_err(function() tonumber(3., 4) end))
end

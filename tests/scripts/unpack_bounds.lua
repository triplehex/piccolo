
local function count_args(...)
    return select("#", ...)
end

do
    local ok, status = pcall(function()
        local _ = table.unpack({}, 1, 1 << 31)
    end)
    assert(not ok)
end

do
    local val = { [-1] = 1, [0] = 2, 3, 4 }

    local first = -1
    local a, b, c, d = table.unpack(val, first)
    assert(count_args(table.unpack(val, first)) == 4)
    assert(a == 1 and b == 2 and c == 3 and d == 4)
end

do
    local val = { 1, 2, 3, 4 }

    assert(table.unpack(val, 2, 2) == 2)
    assert(count_args(table.unpack(val, 2, 2)) == 1)

    assert(count_args(table.unpack(val, 3, 2)) == 0)
    assert(count_args(table.unpack(val, 1, -1)) == 0)
end

do
    local val = { 1, 2, 3, 4 }

    setmetatable(val, { __len = function() return 1 end })

    assert(table.unpack(val) == 1)
    assert(count_args(table.unpack(val)) == 1)

    setmetatable(val, { __len = function() return 0 end })

    assert(table.unpack(val) == nil)
    assert(count_args(table.unpack(val)) == 0)

    setmetatable(val, { __len = function() return -1 end })

    assert(table.unpack(val) == nil)
    assert(count_args(table.unpack(val)) == 0)
end

do
    local first = (1 << 63)     -- i64::MIN
    local last  = (1 << 63) - 1 -- i64::MAX
    -- computing length with `end - start + 1` will overflow

    -- check all cases for getting the length:
    local ok, status
    ok, status = pcall(function()
        local val = { 1, 2, 3, 4 }

        local _ = count_args(table.unpack(val, first, last))
    end)
    assert(not ok)

    ok, status = pcall(function()
        local val = setmetatable({ 1, 2, 3, 4 }, {
            __len = function() return (1 << 63) - 1 end
        })
        local _ = count_args(table.unpack(val, first))
    end)
    assert(not ok)

    ok, status = pcall(function()
        local val = setmetatable({ 1, 2, 3, 4 }, {
            __index = function() return 3 end
        })
        local _ = count_args(table.unpack(val, first, last))
    end)
    assert(not ok)
end

-- Note: this test is a bit slow, but should only use 16MiB of RAM
-- for the unpacked values.  (Potentially multiplied by copies?)
do
    local val = {}
    assert(count_args(table.unpack(val, 1, (1 << 20))) == (1 << 20))

    local ok, status = pcall(function()
        local _ = count_args(table.unpack(val, 1, (1 << 20) + 1))
    end)
    assert(not ok)
end

do
    local val = setmetatable({ }, {
        __len = function() return (1 << 20) end
    })
    assert(count_args(table.unpack(val)) == (1 << 20))

    local ok, status = pcall(function()
        val = setmetatable({ }, {
            __len = function() return (1 << 20) + 1 end
        })
        local _ = count_args(table.unpack(val))
    end)
    assert(not ok)
end

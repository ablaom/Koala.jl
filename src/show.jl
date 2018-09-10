const SHOW_DEPTH = 2
const COLUMN_WIDTH = 25

# short version of showing a `BaseType` object:
function Base.show(stream::IO, object::BaseType)
    abbreviated(n) = "..."*string(n)[end-2:end]
    print(stream, string(typeof(object).name.name,
                         "@", abbreviated(hash(object))))
end

# long version of showing a `BaseType` object:
function Base.show(stream::IO, ::MIME"text/plain", object::BaseType)
    _showall(stream, object, 1)
end

# The above _showall method (defined at the end of this file) will
# generate a table of the field values of the `BaseType` object,
# dislaying each value by calling the method `_show` on it. The
# behaviour of `_show(stream, f)` is a s follows:
#
# 1. If `f` is itself a `BaseType` object, then its short form
# is shown, with a separate table for its own field values appearing
# subsequently (and so on, up to a depth of SHOW_DEPTH).
#
# 2. Otherwise `f` is displayed as "omitted T" where `T =
# typeof(f)`, unless `istoobig(f)` is false (the `istoobig` fall-back
# for arbitrary types being `true`). In the latter case, the long (ie,
# MIME"plain/text") form of `f` is shown. To override this behaviour,
# overload the `_show` method for the type in question.

istoobig(::Any) = true
istoobig(::Number) = false
istoobig(x::AbstractArray) = maximum(size(x)) > 5 

# _show fallback:
function _show(stream::IO, object)
    if !istoobig(object)
        show(stream, MIME("text/plain"), object)
        println(stream)
    else
        println(stream, "omitted ", typeof(object))
    end
end

# _show for BaseType:
_show(stream::IO, object::BaseType) = println(stream, object)

# _show for other types:
function _show(stream::IO, df::DataFrame)
    println(stream, "omitted DataFrame of size $(size(df))")
end

function _show(stream::IO, A::Array{T}) where T
    if !istoobig(A)
        show(stream, MIME("text/plain"), A)
        println(stream)
    else
        println(stream, "omitted Array{$T} of size $(size(A))")
    end
end

function _show(stream::IO, v::Array{T, 1}) where T
    if !istoobig(v)
        show(stream, MIME("text/plain"), v)
        println(stream)
    else
        println(stream, "omitted Vector{$T} of length $(length(v))")
    end
end

# method for generating a field table for BaseType objects:
function _showall(stream::IO, object::BaseType, depth)
    print(stream, "#"^depth, " ")
    show(stream, object)
    println(stream, ": ")
    println(stream, "````")
    names = fieldnames(typeof(object))
    for fld in names
        fld_string = string(fld)*" "^(max(0,COLUMN_WIDTH - length(string(fld))))*"=>   "
        print(stream, fld_string)
        if isdefined(object, fld)
            _show(stream, getfield(object, fld))
 #           println(stream)
        else
            println(stream, "(undefined)")
 #           println(stream)
        end
    end
    println(stream, "````")
    if depth < 2
        for fld in names
            if isdefined(object, fld)
                subobject = getfield(object, fld)
                if isa(subobject, BaseType)
                    _showall(stream, getfield(object, fld), depth + 1)
                end
            end
        end
    end
end




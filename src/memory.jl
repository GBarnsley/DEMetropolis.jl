#no memory
struct DifferentialEvolutionMemoryless{T} <: AbstractDifferentialEvolutionMemory{T}
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

#abstract memory-full type
abstract type AbstractDifferentialEvolutionMemoryFormat{
    T, VV <: AbstractVector{<:AbstractVector{T}},
} <:
AbstractDifferentialEvolutionMemory{T} end

#full memory no changes
struct DifferentialEvolutionMemoryFull{T, VV <: AbstractVector{<:AbstractVector{T}}} <:
    AbstractDifferentialEvolutionMemoryFormat{T, VV}
    mem_x::VV
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DifferentialEvolutionMemoryFull{T, VV},
        x::VV
    ) where {T <: Real, VV <: AbstractVector{<:AbstractVector{T}}}
    return memory
end

#abstract type for methods of filling memory
abstract type AbstractDifferentialEvolutionMemoryFillMethod end

#full memory, refreshing from the start
struct DifferentialEvolutionMemoryRefill{T, VV <: AbstractVector{<:AbstractVector{T}}} <:
    AbstractDifferentialEvolutionMemoryFormat{T, VV}
    mem_x::VV
    fill::AbstractDifferentialEvolutionMemoryFillMethod
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DifferentialEvolutionMemoryRefill{T, VV},
        x::VV
    ) where {T <: Real, VV <: AbstractVector{<:AbstractVector{T}}}
    if update_position!(memory.fill)
        @inbounds for i in 1:memory.fill.n_chains
            memory.mem_x[memory.fill.position - i + 1] .= x[i]
        end
    end
    if memory.fill.position == length(memory.mem_x)
        memory.fill.position = 0
    end
    return memory
end

#non-full memory, filling up to a max size
struct DifferentialEvolutionMemoryFill{T, VV <: AbstractVector{<:AbstractVector{T}}} <:
    AbstractDifferentialEvolutionMemoryFormat{T, VV}
    mem_x::VV
    fill::AbstractDifferentialEvolutionMemoryFillMethod
    refill::Bool
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DifferentialEvolutionMemoryFill{T, VV},
        x::VV
    ) where {T <: Real, VV <: AbstractVector{<:AbstractVector{T}}}
    if update_position!(memory.fill)
        @inbounds for i in 1:memory.fill.n_chains
            memory.mem_x[memory.fill.position - i + 1] .= x[i]
        end
    end

    if memory.fill.position == length(memory.mem_x)
        if memory.refill
            memory = DifferentialEvolutionMemoryRefill(memory.mem_x, memory.fill, memory.indices_INTERNAL, memory.ordered_indices_INTERNAL)
            memory.fill.position = 0
        else
            memory = DifferentialEvolutionMemoryFull(memory.mem_x, memory.indices_INTERNAL, memory.ordered_indices_INTERNAL)
        end
    end
    return memory
end

mutable struct DifferentialEvolutionMemoryFillEvery <:
    AbstractDifferentialEvolutionMemoryFillMethod
    position::Int
    n_chains::Int
end

function update_position!(method::DifferentialEvolutionMemoryFillEvery)
    method.position += method.n_chains
    return true
end

mutable struct DifferentialEvolutionMemoryFillThin <:
    AbstractDifferentialEvolutionMemoryFillMethod
    position::Int
    n_chains::Int
    count::Int
    max_count::Int
end

function update_position!(method::DifferentialEvolutionMemoryFillThin)
    method.count -= 1
    if method.count == 0
        method.position += method.n_chains
        method.count = method.max_count
        return true
    else
        return false
    end
end

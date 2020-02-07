# generate colored noises

using Base.Iterators: product

using WAV, DSP, NPZ, PyCall

# make colored filter cofficients (n taps)
# ref: https://jp.mathworks.com/help/dsp/ref/dsp.colorednoise-system-object.html
function arcoffs(α, n)
  coffs = zeros(Float64, n+1)

  for k in 0:n
    if k == 0
      coffs[k+1] = 1.0
      continue
    end

    coffs[k+1] = (k - 1 + α/2) * coffs[k] / k
  end

  coffs
end

# normalize by dividing max amplitude
function normalize(xs)
  m = maximum(abs2.(xs))
  return xs ./ sqrt(m)
end

# generate filtered (or not filtered) noise
function noise_gen(fs, sec, filter=nothing)
  nsamples = fs * sec

  # white noise
  noise = randn(nsamples)
  noise = filter === nothing ? noise : filt(filter, noise)

  normalize(noise)
end

function str2sym(d::Dict{String, T}) where T
  Dict(Symbol(k) => v for (k, v) in collect(d))
end

function numpysavez(path, dict)
  numpy = pyimport("numpy")
  numpy.savez(path; str2sym(dict)...)
end

function main()
  # ESC-50-like setting
  fs  = 16000
  sec = 5

  # the number of fold
  nfolds = 5

  # waves per class
  ninstances = 40

  # labels
  nlabels = 5

  # label to alpha
  label2filter = Dict(
    0 => FIRFilter(arcoffs(-2, 63)),  # violet
    1 => FIRFilter(arcoffs(-1, 63)),  # blue
    2 => nothing,                     # white
    3 => FIRFilter(arcoffs( 1, 63)),  # pink
    4 => FIRFilter(arcoffs( 2, 63)),  # brawn
  )

  noise_dataset = Dict{String, Dict{String, Any}}()

  for fold in 1:nfolds
    noise_dataset["fold" * string(fold)] = Dict{String, Any}()

    samples = Array{Float64}[]
    labels  = Int64[]

    for (label, i) in product(0:nlabels-1, 1:ninstances)
      push!(samples, noise_gen(fs, sec, label2filter[label]))
      push!(labels, label)
    end

    noise_dataset["fold" * string(fold)]["sounds"] = samples
    noise_dataset["fold" * string(fold)]["labels"] = labels
  end

  mkpath("data/noises")

  numpysavez("data/noises/wav16.npz", noise_dataset)
end

main()
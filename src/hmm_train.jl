using ArgParse
using Pickle
using CUDA
using CSV
using DataFrames
using ProbabilisticCircuits
using ProbabilisticCircuits: PlainInputNode, PlainSumNode, CuBitsProbCircuit,
multiply, loglikelihood, full_batch_em, update_parameters


function dataset_cpu(dataset_path, sample_length; padding=true)
    dataframe = CSV.read(dataset_path, DataFrame;
        header=false, types=Union{UInt32, Missing}, strict=true)
    if padding
        m = map(x -> x==50257 ? UInt32(50256) : x, Tables.matrix(dataframe))
    else
        m = map(x -> x==50257 ? missing : x, Tables.matrix(dataframe))
    end
    return m[:, 1:sample_length]
end


function init()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--cuda_id"
            help = "CUDA ID"
            arg_type = Int64
            default = 1
        "--model_path"
            help = "path for saving/loading checkpoints"
            arg_type = String
            default = ""
        "--checkpoint"
            help = "start iterations"
            arg_type = Int64
            default = 0
        "--max_epochs"
            help = "max iterations"
            arg_type = Int64
            default = 1000
        "--train_data_file"
            help = "train file path"
            arg_type = String
        "--sample_length"
            help = ""
            arg_type = Int64
        "--hidden_states"
            help = "number of clusters used in warmup"
            arg_type = Int64
            default = 512
        "--vocab_size"
            help = ""
            arg_type = Int64
            default = 50257
        "--batch_size"
            help = "batch_size"
            arg_type = Int64
            default = 512
        "--pseudocount"
            help = "pseudocount"
            arg_type = Float64
            default = 1.0
        "--log_file"
            help = "log file"
            arg_type = String
    end
    args = parse_args(ARGS, s)

    args
end


function load_hmm(checkpoint_file, sample_length)
    
    x = Pickle.Torch.THload(checkpoint_file)
    hidden_states = x["hidden_states"]
    vocab_size = x["vocab_size"]
    alpha = x["alpha"]
    beta = x["beta"]
    gamma = x["gamma"]

    input2group = Dict()
    sum2group = Dict()

    layer = Any[]
    inputs = Any[]

    for suffix_len in 1:sample_length
        var = sample_length - suffix_len + 1

        # construct input nodes
        inputs = Any[]
        for u in 1:hidden_states
            weights = beta[u, :]
            input = PlainInputNode(var, Categorical(weights))
            push!(inputs, input)
        end

        for u in 1:hidden_states
            input2group[inputs[u]] = u
        end

        # construct linear layer
        if suffix_len == 1
            layer = inputs
        else
            layer_new = Any[]
            for u in 1:hidden_states
                children = [layer[v] for v in 1:hidden_states]
                sum_node = PlainSumNode(children, alpha[u, :])
                sum2group[sum_node] = u
                push!(layer_new, multiply(inputs[u], sum_node))
            end
            layer = layer_new
        end
    end

    pc = PlainSumNode(layer, gamma)

    pc2hmm = Dict(
        "state2sum" => [x.inputs[2] for x in layer],
        "state2input" => inputs,
    )

    pc, input2group, sum2group, hidden_states, vocab_size, pc2hmm
end


function save_hmm(pc, pc2hmm, hidden_states, vocab_size,
    checkpoint_file_path)

    state2sum, state2input = pc2hmm["state2sum"], pc2hmm["state2input"]

    # write alpha
    alpha = Array{Float32}(undef, hidden_states, hidden_states)
    beta = Array{Float32}(undef, hidden_states, vocab_size)
    for u in 1:hidden_states
        alpha[u, :] = state2sum[u].params
    end

    # write beta
    for u in 1:hidden_states
        beta[u, :] = state2input[u].dist.logps
    end

    # write gamma
    gamma = Float32.(pc.params)

    Pickle.Torch.THsave(checkpoint_file_path, Dict(
        "hidden_states" => hidden_states,
        "vocab_size" => vocab_size,
        "alpha" => alpha,
        "beta" => beta,
        "gamma" => gamma,
    ))
end


function train_hmm(bpc, node2group, edge2group,
    checkpoint, max_epochs, batch_size, pseudocount,
    model_path, pc, pc2hmm, hidden_states, vocab_size,
    train_data_file, sample_length, log_file)

    for epoch in checkpoint+1:max_epochs
        load_path = "$train_data_file.$epoch"
        println("loading train data $load_path ...")
        data_epoch = dataset_cpu(load_path, sample_length)
        data_size = size(data_epoch)[1]

        validation_epoch = cu(data_epoch[1:div(data_size, 10), :])
        train_epoch = cu(data_epoch[div(data_size, 10)+1:data_size, :])

        if epoch == checkpoint+1
            ll = loglikelihood(bpc, validation_epoch; batch_size)
            println("$(checkpoint)\t0.0\t$(ll)")
            open(log_file, "a+") do fout
                write(fout, "$(checkpoint)\t0.0\t$(ll)\n")
            end
        end

        println("Full batch epoch = ", epoch)
        @time train_ll = full_batch_em(bpc, train_epoch, 1;
            batch_size, pseudocount, node2group, edge2group)

        validation_ll = loglikelihood(bpc, validation_epoch; batch_size)
        println("$(epoch)\t$(train_ll[end])\t$(validation_ll)")
        open(log_file, "a+") do fout
            write(fout, "$(epoch)\t$(train_ll[end])\t$(validation_ll)\n")
        end

        println("Free memory")
        @time begin
            CUDA.unsafe_free!(train_epoch)
            CUDA.unsafe_free!(validation_epoch)
        end

        # save checkpoint every 5 epoch
        if !isnothing(model_path) && model_path!= "" && epoch % 5 == 0
            update_parameters(bpc)
            checkpoint_path = model_path * "/checkpoint-$(epoch).weight.th"
            save_hmm(pc, pc2hmm, hidden_states, vocab_size,
                checkpoint_path)
        end
    end
end


function main()
    args = init()

    println(args)
    device!(args["cuda_id"])

    open(args["log_file"], "a+") do fout
        write(fout, join(ARGS, " ") * "\n")
    end

    # load checkpoint
    checkpoint_file = args["model_path"] * "/" * "checkpoint-$(args["checkpoint"]).weight.th"
    println("loading params from $(checkpoint_file) ...")
    @time pc, input2group, sum2group, hidden_states, vocab_size, pc2hmm = load_hmm(
        checkpoint_file, args["sample_length"])

    println("gc ...")
    @time GC.gc()

    println("moving circuit to gpu ...")
    CUDA.@time bpc, node2group, edge2group = CuBitsProbCircuit(pc, input2group, sum2group)

    println("gc ...")
    @time GC.gc()

    @time println("training hmm with $(num_parameters(pc)) params and $(num_nodes(pc)) nodes ...")

    # free memory
    println("runing gc to free RAM ...")
    input2group, sum2group = nothing, nothing
    @time GC.gc()

    println("training hmm ...")
    train_hmm(bpc, node2group, edge2group,
        args["checkpoint"], args["max_epochs"], args["batch_size"], args["pseudocount"],
        args["model_path"], pc, pc2hmm, hidden_states, vocab_size,
        args["train_data_file"], args["sample_length"], args["log_file"])

    println()
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
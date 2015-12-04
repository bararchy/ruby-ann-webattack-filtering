require 'stopwords'
require 'fast_stemmer'
require 'narray'
require 'webrick'

class BagOfWords
  attr_reader :term_index, :doc_hashes, :doc_count, :doc_frequency

  DEFAULT_OPTS = {
    # tf: :raw,
    idf: false,
    stem: true,
    min_term_length: 1
  }

  def initialize opts = {}
    @opts = DEFAULT_OPTS.merge opts
    @index = 0
    @doc_count = 0
    @term_index = Hash.new do |hsh, key|
      val = hsh[key] = @index
      @index += 1
      val
    end
    @doc_frequency = Hash.new(0)
    @doc_hashes = []
  end

  def terms_count
    @index
  end

  def add_docs docs
    docs.each {|doc| add_doc(doc) }
  end

  def add_doc doc
    @doc_count += 1
    terms = extract_terms doc
    doc_hash = create_raw_doc_hash terms
    normalize_tf doc_hash
    # update_doc_hash_tf(doc_hash) unless @opts[:tf] == :raw
    update_doc_frequency doc_hash
    @doc_hashes << doc_hash
  end

  def to_a
    apply_idf_weighting! if @opts[:idf] && !@idf_weighting_applied

    @doc_hashes.map do |doc_hash|
      vec = Array.new(@index, 0)

      doc_hash.each do |k, v|
        vec[k] = v
      end

      vec
    end
  end

  def to_matrix float_precision = :double
    apply_idf_weighting! if @opts[:idf] && !@idf_weighting_applied

    typecode = { double: NArray::DFLOAT, single: NArray::SFLOAT }[float_precision]
    matrix = NMatrix.new(typecode, terms_count, doc_count)

    @doc_hashes.each_with_index do |doc_hash, i|
      doc_hash.each do |k, v|
        matrix[k, i] = v
      end
    end

    matrix
  end

  private

    def apply_idf_weighting!
      @doc_hashes.each do |doc_hash|
        doc_hash.each do |k, v|
          idf = calculate_idf k
          doc_hash[k] = v * idf
        end
      end

      @idf_weighting_applied = true
    end

    def extract_terms doc
      doc.scrub!
      terms = doc.downcase.strip.split(/\s/)
      terms.reject! {|t| t.length < @opts[:min_term_length]} if @opts[:min_term_length]
      terms.map! {|t| t.stem} if @opts[:stem]
      terms
    end

    def create_raw_doc_hash terms
      out = terms.inject(Hash.new(0)) do |hsh, term|
        index = @term_index[term]
        hsh[index] +=1
        hsh
      end
    end

    def normalize_tf doc_hash
      normalize_data = (doc_hash.values.map {|v| v**2}.reduce(:+))
      unless normalize_data.nil?
        norm = Math.sqrt normalize_data
        doc_hash.each do |k, v|
          doc_hash[k] = v / norm
        end
      end
    end

    def update_doc_frequency doc_hash
      doc_hash.each_key do |k|
        @doc_frequency[k] += 1
      end
    end

    def calculate_idf term_id
      Math.log (@doc_count / @doc_frequency[term_id].to_f)
    end
end


class NeuralNet
  attr_reader :shape, :outputs
  attr_accessor :weights, :weight_update_values

  DEFAULT_TRAINING_OPTIONS = {
    max_iterations:   1_000,
    error_threshold:  0.01
  }

  def initialize shape
    @shape = shape
  end

  def run input
    # Input to this method represents the output of the first layer (i.e., the input layer)
    @outputs = [input]
    set_initial_weight_values if @weights.nil?

    # Now calculate output of neurons in subsequent layers:
    1.upto(output_layer).each do |layer|
      source_layer = layer - 1 # i.e, the layer that is feeding into this one
      source_outputs = @outputs[source_layer]

      @outputs[layer] = @weights[layer].map do |neuron_weights|
        # inputs to this neuron are the neuron outputs from the source layer times weights
        inputs = neuron_weights.map.with_index do |weight, i|
          source_output = source_outputs[i] || 1 # if no output, this is the bias neuron
          weight * source_output
        end

        sum_of_inputs = inputs.reduce(:+)
        # the activated output of this neuron (using sigmoid activation function)
        sigmoid sum_of_inputs
      end
    end

    # Outputs of neurons in the last layer is the final result
    @outputs[output_layer]
  end

  def train inputs, expected_outputs, opts = {}
    opts = DEFAULT_TRAINING_OPTIONS.merge(opts)
    error_threshold, log_every = opts[:error_threshold], opts[:log_every]
    iteration, error = 0, 0

    set_initial_weight_update_values if @weight_update_values.nil?
    set_weight_changes_to_zeros
    set_previous_gradients_to_zeroes

    while iteration < opts[:max_iterations]
      iteration += 1

      error = train_on_batch(inputs, expected_outputs)

      if log_every && (iteration % log_every == 0)
        puts "[#{iteration}] #{(error * 100).round(2)}% mse"
      end

      break if error_threshold && (error < error_threshold)
    end

    {error: error.round(5), iterations: iteration, below_error_threshold: (error < error_threshold)}
  end

  private

    def train_on_batch inputs, expected_outputs
      total_mse = 0

      set_gradients_to_zeroes

      inputs.each.with_index do |input, i|
        run input
        training_error = calculate_training_error expected_outputs[i]
        update_gradients training_error
        total_mse += mean_squared_error training_error
      end

      update_weights

      total_mse / inputs.length.to_f # average mean squared error for batch
    end

    def calculate_training_error ideal_output
      @outputs[output_layer].map.with_index do |output, i| 
        output - ideal_output[i]
      end
    end

    def update_gradients training_error
      deltas = {}
      # Starting from output layer and working backwards, backpropagating the training error
      output_layer.downto(1).each do |layer|
        deltas[layer] = []

        @shape[layer].times do |neuron|
          neuron_error = if layer == output_layer
            -training_error[neuron]
          else
            target_layer = layer + 1

            weighted_target_deltas = deltas[target_layer].map.with_index do |target_delta, target_neuron| 
              target_weight = @weights[target_layer][target_neuron][neuron]
              target_delta * target_weight
            end

            weighted_target_deltas.reduce(:+)
          end

          output = @outputs[layer][neuron]
          activation_derivative = output * (1.0 - output)

          delta = deltas[layer][neuron] = neuron_error * activation_derivative

          # gradient for each of this neuron's incoming weights is calculated:
          # the last output from incoming source neuron (from -1 layer)
          # times this neuron's delta (calculated from error coming back from +1 layer)
          source_neurons = @shape[layer - 1] + 1 # account for bias neuron
          source_outputs = @outputs[layer - 1]
          gradients = @gradients[layer][neuron]

          source_neurons.times do |source_neuron|
            source_output = source_outputs[source_neuron] || 1 # if no output, this is the bias neuron
            gradient = source_output * delta
            gradients[source_neuron] += gradient # accumulate gradients from batch
          end
        end
      end
    end

    MIN_STEP, MAX_STEP = Math.exp(-6), 50

    # Now that we've calculated gradients for the batch, we can use these to update the weights
    # Using the RPROP algorithm - somewhat more complicated than classic backpropagation algorithm, but much faster
    def update_weights
      1.upto(output_layer) do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # account for bias neuron

        @shape[layer].times do |neuron|
          source_neurons.times do |source_neuron|
            weight_change = @weight_changes[layer][neuron][source_neuron]
            weight_update_value = @weight_update_values[layer][neuron][source_neuron]
            # for RPROP, we use the negative of the calculated gradient
            gradient = -@gradients[layer][neuron][source_neuron]
            previous_gradient = @previous_gradients[layer][neuron][source_neuron]

            c = sign(gradient * previous_gradient)

            case c
              when 1 then # no sign change; accelerate gradient descent
                weight_update_value = [weight_update_value * 1.2, MAX_STEP].min
                weight_change = -sign(gradient) * weight_update_value
              when -1 then # sign change; we've jumped over a local minimum
                weight_update_value = [weight_update_value * 0.5, MIN_STEP].max
                weight_change = -weight_change # roll back previous weight change
                gradient = 0 # so won't trigger sign change on next update
              when 0 then
                weight_change = -sign(gradient) * weight_update_value
            end

            @weights[layer][neuron][source_neuron] += weight_change
            @weight_changes[layer][neuron][source_neuron] = weight_change
            @weight_update_values[layer][neuron][source_neuron] = weight_update_value
            @previous_gradients[layer][neuron][source_neuron] = gradient
          end
        end
      end
    end

    def set_weight_changes_to_zeros
      @weight_changes = build_connection_matrixes { 0.0 }
    end

    def set_gradients_to_zeroes
      @gradients = build_connection_matrixes { 0.0 }
    end

    def set_previous_gradients_to_zeroes
      @previous_gradients = build_connection_matrixes { 0.0 }
    end

    def set_initial_weight_update_values
      @weight_update_values = build_connection_matrixes { 0.1 }
    end

    def set_initial_weight_values
      # Initialize all weights to random float value
      @weights = build_connection_matrixes { rand(-0.5..0.5) }

      # Update weights for first hidden layer (Nguyen-Widrow method)
      # This is a bit obscure, and not entirely necessary, but it should help the network train faster
      beta = 0.7 * @shape[1]**(1.0 / @shape[0])

      @shape[1].times do |neuron|
        weights = @weights[1][neuron]
        norm = Math.sqrt weights.map {|w| w**2}.reduce(:+)
        updated_weights = weights.map {|weight| (beta * weight) / norm }
        @weights[1][neuron] = updated_weights
      end
    end

    def build_connection_matrixes
      1.upto(output_layer).inject({}) do |hsh, layer|
        # Number of incoming connections to each neuron in this layer:
        source_neurons = @shape[layer - 1] + 1 # == number of neurons in prev layer + a bias neuron

        # matrix[neuron] == Array of values for each incoming connection to neuron
        matrix = Array.new(@shape[layer]) do |neuron|
          Array.new(source_neurons) { yield }
        end

        hsh[layer] = matrix
        hsh
      end
    end

    def output_layer
      @shape.length - 1
    end

    def sigmoid x
      1 / (1 + Math.exp(-x))
    end

    def mean_squared_error errors
      errors.map {|e| e**2}.reduce(:+) / errors.length.to_f
    end

    ZERO_TOLERANCE = Math.exp(-16)

    def sign x
      if x > ZERO_TOLERANCE
        1
      elsif x < -ZERO_TOLERANCE
        -1
      else
        0 # x is zero, or a float very close to zero
      end
    end

    def marshal_dump
      [@shape, @weights, @weight_update_values]
    end

    def marshal_load array
      @shape, @weights, @weight_update_values = array
    end
end

module Scaler
  def self.mean data
    data.mean(1)
  end

  def self.std data
    std = data.stddev(1)
    std[std.eq(0)] = 1.0 # so we don't divide by 0
    std
  end

  def self.scale data, mean = nil, std = nil, typecode = nil
    data = NArray.ref(data)
    mean ||= self.mean(data)
    std ||= self.std(data)
    data = (data - mean) / std
    [NMatrix.ref(data), mean, std]
  end

  def self.row_norms data
    squared_data = NArray.ref(data)**2
    NMatrix.ref(squared_data).sum(0)
  end
end



## Option One, Need to get more info regarding this:

bag = BagOfWords.new idf: true
File.open("testing.txt").each do |line|
  doc = URI.decode(line.split(",")[0].chomp.to_s)
  bag.add_doc doc
end

nn = NeuralNet.new [bag.terms_count, 50, 3]

rows = File.readlines("testing.txt").map {|l| l.chomp.split(',') }

label_encodings = {
  "Normal"     => [1, 0, 0],
  "SQLi" => [0, 1, 0],
  "XSS"  => [0, 0 ,1]
}

x_data = bag.to_a
y_data = rows.map {|row| label_encodings[row[1]] }


x_train = x_data.slice(0, 17)
y_train = y_data.slice(0, 17)

x_test = x_data.slice(17, 0)
y_test = y_data.slice(17, 0)



prediction_success = -> (actual, ideal) {
  predicted = (0..2).max_by {|i| actual[i] }
  ideal[predicted] == 1
}

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

error_rate = -> (errors, total) { ((errors / total.to_f) * 100) }

run_test = -> (nn, inputs, expected_outputs) {
  success, failure, errsum = 0,0,0
  inputs.each.with_index do |input, i|
    output = nn.run input
    prediction_success.(output, expected_outputs[i]) ? success += 1 : failure += 1
    errsum += mse.(output, expected_outputs[i])
  end
  [success, failure, errsum / inputs.length.to_f]
}

puts "Testing the untrained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Untrained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"


puts "\nTraining the network...\n\n"

t1 = Time.now
result = nn.train(x_train, y_train, error_threshold: 0.01, 
                                    max_iterations: 1_000,
                                    log_every: 100
                                    )

# puts result
puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).round(1)}s"


puts "\nTesting the trained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"



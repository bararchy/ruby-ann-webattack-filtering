require 'stopwords'
require 'fast_stemmer'
require 'narray'
require 'webrick'
require 'colorize'
require_relative "lib/bag_of_words.rb"
require_relative "lib/rann.rb"


bag = BagOfWords.new idf: true # Initialize the bag


# Load the files to the bag instance
File.open("data/abnormal_http_traffic.txt").each do |line|
  doc = URI.decode(line.scrub.split(",")[0].chomp.to_s)
  bag.add_doc doc
end

File.open("data/normal_http_traffic.txt").each do |line|
  doc = (line.scrub.split(",")[0].chomp.to_s)
  bag.add_doc doc
end

# Initialize the Neural Net, [input neurons, hidden neurons, output neurons]
nn = NeuralNet.new [bag.terms_count, 50, 2]

# Load all the files again to create the y_data object (labels and line numbers)
rows = File.readlines("data/abnormal_http_traffic.txt").map {|l| l.chomp.split(',') }
rows = rows + File.readlines("data/normal_http_traffic.txt").map {|l| l.chomp.split(',') }

# Suffle so it will no be ordered
rows.shuffle!

# Set labels to understand the output options
label_encodings = {
  "Normal"     => [1, 0, 0],
  "Abnormal" => [0, 1, 0]
}

x_data = bag.to_a
y_data = rows.map {|row| label_encodings[row[1]] }


x_train = x_data.slice(0, rows.size)
y_train = y_data.slice(0, rows.size)

# In a better example for this POC another dataset should be tested against with different data.
x_test = x_train
y_test = y_train

puts "Data size: x = #{x_train.size} y = #{y_train.size}"


prediction_success = -> (actual, ideal) {
  predicted = (0..1).max_by {|i| actual[i] }
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




puts "Testing the untrained network...".bold

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts ["Untrained classification success: ","#{success},".green, "failure: ","#{failure}".red, "(classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"].join


puts "\nTraining the network...\n\n".bold

t1 = Time.now
result = nn.train(x_train, y_train, error_threshold: 0.01, max_iterations: 1_000, log_every: 1)

# puts result
puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).round(1)}s"


puts "\nTesting the trained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"

puts "Testing Single input"

new_bag = BagOfWords.new idf: true
new_bag.add_doc "index.html?id=12'OR 'a'='a' SELECT FROM DATA"

xtrain = new_bag.to_a.slice(0,1)
puts xtrain.inspect

output = nn.run xtrain[0]
puts output.inspect
predicted = (0..1).max_by {|i| output[i] }
puts predicted
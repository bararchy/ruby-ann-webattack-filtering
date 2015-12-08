require 'stopwords'
require 'fast_stemmer'
require 'narray'
require 'webrick'
require 'colorize'
require_relative "lib/bag_of_words.rb"
require_relative "lib/rann.rb"


bag = BagOfWords.new idf: false, stem: false # Initialize the bag


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
# if File.exist?("/tmp/ann.db")
#   puts "Loading saved NN from file".bold.green
#   nn = Marshal.load(File.read("/tmp/ann.db"))
# else
  puts "Initializing a new NN instance".bold.green
  nn = NeuralNet.new [bag.terms_count, 50, 2]
# end
# Load all the files again to create the y_data object (labels and line numbers)
rows = File.readlines("data/abnormal_http_traffic.txt").map {|l| l.chomp.split(',') }
rows = rows + File.readlines("data/normal_http_traffic.txt").map {|l| l.chomp.split(',') }

# Set labels to understand the output options
label_encodings = {
  "Normal"     => [1,0],
  "Abnormal" => [0,1]
}

x_data = bag.to_a
y_data = rows.map {|row| label_encodings[row[1]] }

shuffled = x_data.zip(y_data).shuffle

x_data = shuffled.map {|row| row[0]}
y_data = shuffled.map {|row| row[1]}

test_set_size = 50
training_set_size = rows.size - test_set_size

x_train = x_data.slice(0, training_set_size)
y_train = y_data.slice(0, training_set_size)

# In a better example for this POC another dataset should be tested against with different data.
x_test = x_data.slice(training_set_size, rows.size)
y_test = y_data.slice(training_set_size, rows.size)

puts "Data size: x = #{x_train.size} y = #{y_train.size}"
puts "Test set makeup:" + y_test.inject(Hash.new(0)) {|hsh, o| hsh[o[0]] +=1; hsh}.inspect


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

puts ["Trained classification success: ","#{success},".green, "failure: ","#{failure}".red, "(classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"].join

# TODO: Need to modify the bag so that you can tokenize a new example using the exact same word mapping
# puts "Testing Single input"

# new_bag = BagOfWords.new idf: true
# new_bag.add_doc URI.decode("GET /Safe-T/Webskin/DefaultWebskin/Tools/bootstrap-3.1.1/css/bootstrap.min.css HTTP/1.1Host: %3B %20 SELECT %20 SLEEP %28 5 %29 &#45;- %20User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:42.0) Gecko/20100101 Firefox/42.0Accept: text/css*/*;q=0.1Accept-Language: en-USen;q=0.5Accept-Encoding: gzip deflateDNT: 1Referer: https://192.168.4.109/Safe-T/Webskin/DefaultWebskin/Style/Main.cssCookie: clz02j6mgdqpzoey8i5010pz; SFTSId=dnsm0flp3iimy0oxjttymdys; .safeApiAuth=6AF734B51DC559EAB8F898DAB445D97910657850CFE6B9A2FEDC71F3468A4D3C51431F27F051215A2A743D6E854DC58AA0E0BE984E0A964A6802D81103FBAE4341BF760022F8309B8E7929C983B1B1D861D13572293091CB062ECAFB35F3CBF7ABBCC258C1031923DDA0ACD390CE0BB185E7F892; __tempRemoteIdentifier__=409a1cb3549f4f8b9f13430035208736Connection: keep-alive")
# ytrain = [[1,0]]
# xtrain = new_bag.to_a.slice(0,1)
# success, failure, avg_mse = run_test.(nn, xtrain, ytrain)
# puts "Success: #{success}, Failure: #{failure}"

puts "Saving State at /tmp/ann.db"
f = File.open("/tmp/ann.db", 'w')
f.write(Marshal.dump(nn))
f.close
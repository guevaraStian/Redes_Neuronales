
# En el siguiente codigo se muestra la creacion de una red neuronal con lenguaje ruby
# Con perceptrones que indican las iteraciones de la red neuronal 
require 'csv'

# Funciones de activación
def sigmoid(x)
  1.0 / (1.0 + Math.exp(-x))
end

def sigmoid_derivative(x)
  x * (1 - x)
end

# Normalización simple entre 0 y 1
def normalize(data)
  data.transpose.map do |feature|
    min, max = feature.min, feature.max
    feature.map { |x| (x - min) / (max - min) }
  end.transpose
end

# Convertir etiquetas a vectores one-hot
def one_hot(label)
  case label
  when 'Iris-setosa'     then [1, 0, 0]
  when 'Iris-versicolor' then [0, 1, 0]
  when 'Iris-virginica'  then [0, 0, 1]
  else [0, 0, 0]
  end
end

# Cargar datos desde CSV
def load_data(file)
  inputs, outputs = [], []
  CSV.foreach(file) do |row|
    next if row.size < 5
    inputs << row[0..3].map(&:to_f)
    outputs << one_hot(row[4])
  end
  inputs = normalize(inputs)
  [inputs, outputs]
end

class NeuralNetwork
  def initialize(input_size, hidden_size, output_size, learning_rate = 0.1)
    @input_size = input_size
    @hidden_size = hidden_size
    @output_size = output_size
    @learning_rate = learning_rate

    @weights_input_hidden = Array.new(input_size) { Array.new(hidden_size) { rand(-1.0..1.0) } }
    @weights_hidden_output = Array.new(hidden_size) { Array.new(output_size) { rand(-1.0..1.0) } }
  end

  def train(inputs_list, targets_list, epochs)
    epochs.times do |epoch|
      inputs_list.each_with_index do |inputs, idx|
        # Forward pass
        hidden_inputs = dot(inputs, @weights_input_hidden)
        hidden_outputs = hidden_inputs.map { |x| sigmoid(x) }

        final_inputs = dot(hidden_outputs, @weights_hidden_output)
        final_outputs = final_inputs.map { |x| sigmoid(x) }

        # Backward pass
        targets = targets_list[idx]
        output_errors = targets.zip(final_outputs).map { |t, o| t - o }
        output_deltas = output_errors.zip(final_outputs).map { |e, o| e * sigmoid_derivative(o) }

        hidden_errors = @weights_hidden_output.transpose.map do |weights|
          weights.zip(output_deltas).map { |w, d| w * d }.sum
        end
        hidden_deltas = hidden_errors.zip(hidden_outputs).map { |e, h| e * sigmoid_derivative(h) }

        # Update weights hidden -> output
        @weights_hidden_output.each_with_index do |weights, i|
          weights.each_with_index do |w, j|
            @weights_hidden_output[i][j] += @learning_rate * output_deltas[j] * hidden_outputs[i]
          end
        end

        # Update weights input -> hidden
        @weights_input_hidden.each_with_index do |weights, i|
          weights.each_with_index do |w, j|
            @weights_input_hidden[i][j] += @learning_rate * hidden_deltas[j] * inputs[i]
          end
        end
      end
    end
  end

  def predict(inputs)
    hidden_inputs = dot(inputs, @weights_input_hidden)
    hidden_outputs = hidden_inputs.map { |x| sigmoid(x) }

    final_inputs = dot(hidden_outputs, @weights_hidden_output)
    final_inputs.map { |x| sigmoid(x) }
  end

  private

  def dot(vector, matrix)
    matrix.transpose.map { |col| vector.zip(col).map { |a, b| a * b }.sum }
  end
end

# Cargar datos Iris desde archivo local
inputs, outputs = load_data('BasesDeDatos/iris.csv')

# Inicializar red
nn = NeuralNetwork.new(4, 5, 3)

# Entrenar
nn.train(inputs, outputs, 1000)

# Prueba simple
puts "\nPredicciones:"
inputs.first(5).each_with_index do |sample, i|
  prediction = nn.predict(sample)
  predicted_class = prediction.each_with_index.max_by { |val, _| val }[1]
  real_class = outputs[i].each_with_index.max_by { |val, _| val }[1]
  puts "Real: #{real_class}, Predicho: #{predicted_class}, Confianza: #{prediction.map { |x| x.round(2) }}"
end

# En el siguiente codigo se muestra la creacion de una red neuronal con lenguaje ruby
# Con perceptrones que indican las iteraciones de la red neuronal 

class Perceptron
  attr_reader :weights, :bias

  def initialize(input_size, learning_rate = 0.1)
    @weights = Array.new(input_size) { rand(-1.0..1.0) }
    @bias = rand(-1.0..1.0)
    @learning_rate = learning_rate
  end

  def activate(sum)
    sum > 0 ? 1 : 0
  end

  def predict(inputs)
    sum = @weights.zip(inputs).map { |w, i| w * i }.sum + @bias
    activate(sum)
  end

  def train(training_data, epochs = 10)
    epochs.times do |epoch|
      training_data.each do |inputs, expected|
        prediction = predict(inputs)
        error = expected - prediction

        # Actualizar pesos y sesgo
        @weights = @weights.each_with_index.map do |w, i|
          w + @learning_rate * error * inputs[i]
        end
        @bias += @learning_rate * error
      end
    end
  end
end

# Datos de entrenamiento para la compuerta lógica AND
# Formato: [entradas], salida esperada
training_data = [
  [[0, 0], 0],
  [[0, 1], 0],
  [[1, 0], 0],
  [[1, 1], 1]
]

# Crear y entrenar el perceptrón
perceptron = Perceptron.new(2)
perceptron.train(training_data, 20)

# Probar el perceptrón entrenado
puts "Resultados después del entrenamiento:"
training_data.each do |inputs, _|
  output = perceptron.predict(inputs)
  puts "Entrada: #{inputs.inspect} => Salida: #{output}"
end
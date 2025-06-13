// Ejemplo sencillo de red nuronal con 4 ejemplos de 2 caracteristicas
// Tambien se usaron 2 neuronas para el procesamiento
// Se importan librerias como math para la colicion de neuronas

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Función de activación sigmoide
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivada de la función sigmoide
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Entradas de entrenamiento (4 ejemplos, 2 características)
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	// Salidas esperadas (para función XOR, por ejemplo)
	expectedOutput := []float64{0, 1, 1, 0}

	// Inicialización de pesos aleatorios
	weightsInputHidden := [][]float64{
		{rand.Float64(), rand.Float64()},
		{rand.Float64(), rand.Float64()},
	}
	weightsHiddenOutput := []float64{rand.Float64(), rand.Float64()}

	// Tasa de aprendizaje
	lr := 0.5

	// Entrenamiento
	for epoch := 0; epoch < 10000; epoch++ {
		for i, input := range inputs {
			// Forward pass
			hidden := make([]float64, 2)
			for j := 0; j < 2; j++ {
				hidden[j] = sigmoid(input[0]*weightsInputHidden[0][j] + input[1]*weightsInputHidden[1][j])
			}

			output := sigmoid(hidden[0]*weightsHiddenOutput[0] + hidden[1]*weightsHiddenOutput[1])

			// Error
			err := expectedOutput[i] - output

			// Backpropagation (muy simple)
			deltaOutput := err * sigmoidDerivative(output)

			for j := 0; j < 2; j++ {
				weightsHiddenOutput[j] += lr * deltaOutput * hidden[j]
			}

			for j := 0; j < 2; j++ {
				deltaHidden := deltaOutput * weightsHiddenOutput[j] * sigmoidDerivative(hidden[j])
				weightsInputHidden[0][j] += lr * deltaHidden * input[0]
				weightsInputHidden[1][j] += lr * deltaHidden * input[1]
			}
		}
	}

	// Pronostico luego del entrenmiento
	fmt.Println("Resultados luego del entrenamiento:")
	for _, input := range inputs {
		hidden := make([]float64, 2)
		for j := 0; j < 2; j++ {
			hidden[j] = sigmoid(input[0]*weightsInputHidden[0][j] + input[1]*weightsInputHidden[1][j])
		}
		output := sigmoid(hidden[0]*weightsHiddenOutput[0] + hidden[1]*weightsHiddenOutput[1])
		fmt.Printf("Entrada: %v -> Salida: %.3f\n", input, output)
	}
}

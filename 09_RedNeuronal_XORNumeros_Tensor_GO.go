// Ejemplo sencillo de red nuronal con 4 ejemplos de 2 caracteristicas
// Tambien se usaron 2 neuronas para el procesamiento
// Se importan librerias como math para la colicion de neuronas
// go get -u gorgonia.org/gorgonia
// go get -u gorgonia.org/tensor
package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var xorInputs = []float32{
	0, 0,
	0, 1,
	1, 0,
	1, 1,
}

var xorOutputs = []float32{
	0,
	1,
	1,
	0,
}

func main() {
	rand.Seed(time.Now().UnixNano())

	g := G.NewGraph()

	// Parámetros
	inputSize := 2
	hiddenSize := 4
	outputSize := 1
	learningRate := 0.1

	// Variables de entrada/salida
	x := G.NewMatrix(g,
		tensor.Float32,
		G.WithShape(4, inputSize),
		G.WithName("x"),
		G.WithValue(tensor.New(tensor.WithBacking(xorInputs), tensor.WithShape(4, inputSize))),
	)

	y := G.NewMatrix(g,
		tensor.Float32,
		G.WithShape(4, outputSize),
		G.WithName("y"),
		G.WithValue(tensor.New(tensor.WithBacking(xorOutputs), tensor.WithShape(4, outputSize))),
	)

	// Pesos capa oculta
	w0 := G.NewMatrix(g, tensor.Float32, G.WithShape(inputSize, hiddenSize), G.WithName("w0"), G.WithInit(G.GlorotN(1.0)))
	b0 := G.NewMatrix(g, tensor.Float32, G.WithShape(1, hiddenSize), G.WithName("b0"), G.WithInit(G.Zeroes()))

	// Pesos capa salida
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(hiddenSize, outputSize), G.WithName("w1"), G.WithInit(G.GlorotN(1.0)))
	b1 := G.NewMatrix(g, tensor.Float32, G.WithShape(1, outputSize), G.WithName("b1"), G.WithInit(G.Zeroes()))

	// Feedforward
	l0 := G.Must(G.Add(G.Must(G.Mul(x, w0)), b0))
	l0Act := G.Must(G.Sigmoid(l0))
	l1 := G.Must(G.Add(G.Must(G.Mul(l0Act, w1)), b1))
	pred := G.Must(G.Sigmoid(l1))

	// Pérdida
	cost := G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(pred, y))))))

	// Gradientes
	_, err := G.Grad(cost, w0, b0, w1, b1)
	if err != nil {
		log.Fatal(err)
	}

	// Máquina de ejecución
	vm := G.NewTapeMachine(g, G.BindDualValues(w0, b0, w1, b1))

	// Entrenamiento
	for i := 0; i < 10000; i++ {
		if err := vm.RunAll(); err != nil {
			log.Fatal(err)
		}

		if i%1000 == 0 {
			fmt.Printf("Epoch %d | Loss: %v\n", i, cost.Value())
		}

		// Descenso del gradiente manual
		G.WithLearnRate(learningRate, w0, b0, w1, b1)
		vm.Reset()
	}

	// Mostrar resultados
	fmt.Println("\nPredicciones finales:")
	fmt.Printf("Input\tOutput\n")
	fmt.Printf("0 0\t%.2f\n", predict(g, vm, []float32{0, 0}, w0, b0, w1, b1))
	fmt.Printf("0 1\t%.2f\n", predict(g, vm, []float32{0, 1}, w0, b0, w1, b1))
	fmt.Printf("1 0\t%.2f\n", predict(g, vm, []float32{1, 0}, w0, b0, w1, b1))
	fmt.Printf("1 1\t%.2f\n", predict(g, vm, []float32{1, 1}, w0, b0, w1, b1))
}

// predict evalúa la red neuronal con una entrada dada
func predict(g *G.ExprGraph, vm G.VM, input []float32, w0, b0, w1, b1 *G.Node) float32 {
	// Capa oculta
	x := tensor.New(tensor.WithBacking(input), tensor.WithShape(1, 2))
	l0 := G.Must(G.Add(G.Must(G.Mul(G.NewTensor(g, tensor.Float32, 2, G.WithShape(1, 2), G.WithValue(x)), w0)), b0))
	l0Act := G.Must(G.Sigmoid(l0))
	l1 := G.Must(G.Add(G.Must(G.Mul(l0Act, w1)), b1))
	pred := G.Must(G.Sigmoid(l1))

	machine := G.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	outVal := pred.Value().Data().([]float32)[0]
	machine.Reset()
	return outVal
}

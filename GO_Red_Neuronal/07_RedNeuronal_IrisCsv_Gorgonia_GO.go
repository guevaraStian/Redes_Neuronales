// Ejemplo sencillo de red nuronal con 4 ejemplos de 2 caracteristicas
// Tambien se usaron 2 neuronas para el procesamiento
// Se importan librerias como math para la colicion de neuronas
// go get -u gorgonia.org/gorgonia
// go get -u gorgonia.org/tensor
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gorgonia.org/gorgonia"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Cargar datos del CSV
	features, labels := loadIrisCSV("BasesDeDatos/iris.csv")

	// Dividir en 5 pliegues
	k := 5
	indices := rand.Perm(len(features))
	folds := make([][]int, k)
	for i := 0; i < len(indices); i++ {
		folds[i%k] = append(folds[i%k], indices[i])
	}

	var totalAccuracy float64

	// Validación cruzada
	for i := 0; i < k; i++ {
		trainFeatures, trainLabels, testFeatures, testLabels := splitData(features, labels, folds, i)

		// Normalizar datos
		trainFeatures = normalize(trainFeatures)
		testFeatures = normalize(testFeatures)

		// Crear y entrenar la red neuronal
		model := createModel()
		trainModel(model, trainFeatures, trainLabels)

		// Evaluar el modelo
		accuracy := evaluateModel(model, testFeatures, testLabels)
		totalAccuracy += accuracy
		fmt.Printf("Fold %d: Precisión = %.4f\n", i+1, accuracy)
	}

	// Precisión promedio
	avgAccuracy := totalAccuracy / float64(k)
	fmt.Printf("Precisión promedio: %.4f\n", avgAccuracy)
}

// Función para cargar los datos del archivo CSV
func loadIrisCSV(filename string) ([][]float32, [][]float32) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var features [][]float32
	var labels [][]float32

	for _, row := range rows {
		var input []float32
		for i := 0; i < 4; i++ {
			val, _ := strconv.ParseFloat(row[i], 32)
			input = append(input, float32(val))
		}
		features = append(features, input)

		// Codificar la etiqueta one-hot
		label := oneHotEncode(row[4])
		labels = append(labels, label)
	}

	return features, labels
}

// Función para codificar las etiquetas en formato one-hot
func oneHotEncode(class string) []float32 {
	switch class {
	case "Iris-setosa":
		return []float32{1, 0, 0}
	case "Iris-versicolor":
		return []float32{0, 1, 0}
	case "Iris-virginica":
		return []float32{0, 0, 1}
	default:
		log.Fatalf("Clase desconocida: %s", class)
		return nil
	}
}

// Función para normalizar los datos
func normalize(data [][]float32) [][]float32 {
	// Normalizar entre 0 y 1
	var normalized [][]float32
	for _, row := range data {
		var normRow []float32
		for _, val := range row {
			normRow = append(normRow, val/10) // Suponiendo que los valores están en el rango 0-10
		}
		normalized = append(normalized, normRow)
	}
	return normalized
}

// Función para dividir los datos en entrenamiento y prueba
func splitData(features, labels [][]float32, folds [][]int, foldIndex int) ([][]float32, [][]float32, [][]float32, [][]float32) {
	var trainFeatures, trainLabels, testFeatures, testLabels [][]float32

	// Indices de prueba
	testIndices := folds[foldIndex]

	// Dividir en conjuntos de entrenamiento y prueba
	for i := 0; i < len(features); i++ {
		if contains(testIndices, i) {
			testFeatures = append(testFeatures, features[i])
			testLabels = append(testLabels, labels[i])
		} else {
			trainFeatures = append(trainFeatures, features[i])
			trainLabels = append(trainLabels, labels[i])
		}
	}

	return trainFeatures, trainLabels, testFeatures, testLabels
}

// Función para verificar si un slice contiene un valor
func contains(slice []int, val int) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}

// Función para crear el modelo de la red neuronal
func createModel() *gorgonia.ExprGraph {
	g := gorgonia.NewGraph()

	// Definir las capas de la red neuronal
	// ...

	return g
}

// Función para entrenar el modelo
func trainModel(model *gorgonia.ExprGraph, trainFeatures, trainLabels [][]float32) {
	// Implementar el entrenamiento del modelo
	// ...

}

// Función para evaluar el modelo
func evaluateModel(model *gorgonia.ExprGraph, testFeatures, testLabels [][]float32) float64 {
	// Implementar la evaluación del modelo
	// ...

	return 0.0
}

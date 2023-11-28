using System;
using System.IO;
using System.Linq;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    abstract class Layer
    {
        protected string _name; // наименование слоя

        private string _weightsDirPath; // путь к каталогу весов
        private string _weightsFilePath; // путь к файлу весов

        protected int _neuronsCount; // число нейронов в текущем слое
        protected int _prevNeuronsCount; // число нейронов в предыдущем слое

        protected const double _learningRate = 0.5d;
        protected const double _momentum = 0.05d;
        protected double[,] _lastDeltaWeights;

        private Neuron[] neurons;

        public Neuron[] Neurons { get => neurons; set => neurons = value; }

        public double[] Data
        {
            set
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i].Inputs = value;
                    Neurons[i].Activator();
                }
            }
        }

        protected Layer(int neuronsCount, int prevNeuronsCount, NeuronType neuronType, string name)
        {
            _name = name;
            _neuronsCount = neuronsCount;
            _prevNeuronsCount = prevNeuronsCount;

            _weightsDirPath = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            _weightsFilePath = _weightsDirPath + name + "_memory.csv";

            Neurons = new Neuron[neuronsCount];

            double[,] weights;

            if (File.Exists(_weightsFilePath))
            {
                weights = WeightInitialize(MemoryMode.GET, _weightsFilePath);
            }
            else
            {
                Directory.CreateDirectory(_weightsDirPath);
                weights = WeightInitialize(MemoryMode.INIT, _weightsFilePath);
            }

            _lastDeltaWeights = new double[neuronsCount, neuronsCount + 1];

            for (int i = 0; i < neuronsCount; i++)
            {
                double[] tmpWeights = new double[prevNeuronsCount + 1];
                for (int j = 0; j < prevNeuronsCount + 1; j++)
                {
                    tmpWeights[j] = weights[i, j];
                }
                Neurons[i] = new Neuron(tmpWeights, neuronType);
            }
        }

        public double[,] WeightInitialize(MemoryMode memoryMode, string path)
        {
            char[] delim = new char[] { ';', ' ' }; // разделители слов

            string tmpStr; // временная строка для чтения
            string[] tmpStrWeights; // временный массив строк


            double[,] weights = new double[_neuronsCount, _prevNeuronsCount + 1];

            switch (memoryMode)
            {
                // прочитать веса из файла
                case MemoryMode.GET:

                    tmpStrWeights = File.ReadAllLines(path);

                    for (int i = 0; i < _neuronsCount; i++)
                    {
                        string[] memoryElement = tmpStrWeights[i].Split(delim);

                        for (int j = 1; j < _prevNeuronsCount + 1; j++)
                        {
                            weights[i, j] = double.Parse(memoryElement[j - 1].Replace(',', '.'), 
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                // записать в файл веса
                case MemoryMode.SET:

                    tmpStrWeights = new string[_neuronsCount];

                    for (int i = 0; i < _neuronsCount; i++)
                    {
                        tmpStr = Neurons[i].Weights[0].ToString();
                        for (int j = 1; j < _prevNeuronsCount + 1; j++)
                        {
                            tmpStr += delim[0] + Neurons[i].Weights[j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    
                    break;

                // проинициализировать веса и записать их в файл
                case MemoryMode.INIT:

                    /*
                        1. Веса инициализируются случайными величинами
                        2. Матожидание (среднее значение) всех весов нейрона должно равняться нулю
                         -> Найти среднее значение, его вычесть из всех значений весов
                        3. Среднеквадратическое отклонение случайных величин должно быть = 1
                     */

                    Random random = new Random();

                    double[] tmpArr = new double[_prevNeuronsCount + 1];
                    
                    tmpStrWeights = new string[_neuronsCount];
                    tmpStr = "";

                    for (int i = 0; i < _neuronsCount; i++)
                    {
                        for (int j = 0; j < _prevNeuronsCount + 1; j++)
                        {
                            tmpArr[j] = 0.02 * random.NextDouble() - 0.01;
                        }

                        double tmpRatio = 1.0d / Math.Sqrt(Dispersion(tmpArr) * (_prevNeuronsCount + 1));
                        double tmpShift = Average(tmpArr);

                        weights[i, 0] = (tmpArr[0] - tmpShift) * tmpRatio;
                        tmpStr = weights[i, 0].ToString();

                        for (int j = 1; j < _prevNeuronsCount + 1; j++)
                        {
                            weights[i, j] = (tmpArr[j] - tmpShift) * tmpRatio;
                            tmpStr += delim[0] + weights[i, j].ToString();
                        }

                        tmpStrWeights[i] = tmpStr;
                    }

                    File.WriteAllLines(path, tmpStrWeights);
                    break;
            }

            return weights;
        }

        private double Average(double[] arr)
        {
            return arr.Sum() / arr.Length;
        }

        private double Dispersion(double[] arr)
        {
            double mean = Average(arr);

            double[] squaredDifferences = new double[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                squaredDifferences[i] = Math.Pow(arr[i] - mean, 2);
            }

            double dispersion = squaredDifferences.Sum() / squaredDifferences.Length;
            return dispersion;
        }

        public abstract void Recognize(NeuroNet net, Layer nextLayer);
        public abstract double[] BackwardPass(double[] stuff);
    }
}

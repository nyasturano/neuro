using System;
using System.IO;

namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    abstract class Layer
    {
        protected string name; // наименование слоя

        private string weightsDirPath; // путь к каталогу весов
        private string weightsFilePath; // путь к файлу весов

        protected int neuronsCount; // число нейронов в текущем слое
        protected int prevNeuronsCount; // число нейронов в предыдущем слое

        protected const double learningRate = 0.001d;
        protected const double momentum = 0.05d;
        protected double[,] lastDeltaWeights;

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

        protected Layer(int _neuronsCount, int _prevNeuronsCount, NeuronType _neuronType, string _name)
        {
            name = _name;
            neuronsCount = _neuronsCount;
            prevNeuronsCount = _prevNeuronsCount;

            weightsDirPath = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            weightsFilePath = weightsDirPath + name + "_memory.csv";

            Neurons = new Neuron[neuronsCount];

            double[,] weights;

            if (File.Exists(weightsFilePath))
            {
                weights = WeightInitialize(MemoryMode.GET);
            }
            else
            {
                Directory.CreateDirectory(weightsDirPath);
                weights = WeightInitialize(MemoryMode.INIT);
            }

            lastDeltaWeights = new double[neuronsCount, prevNeuronsCount + 1];

            // заполняем данные о каждом нейроне
            for (int i = 0; i < neuronsCount; i++)
            {
                double[] tmpWeights = new double[prevNeuronsCount + 1];
                for (int j = 0; j < prevNeuronsCount + 1; j++)
                {
                    tmpWeights[j] = weights[i, j];
                }
                Neurons[i] = new Neuron(tmpWeights, _neuronType);
            }
        }

        public double[,] WeightInitialize(MemoryMode memoryMode)
        {
            char[] delim = new char[] { ';', ' ' }; // разделители слов

            string tmpStr; // временная строка для чтения
            string[] tmpStrWeights; // временный массив строк

            double[,] weights = new double[neuronsCount, prevNeuronsCount + 1];

            switch (memoryMode)
            {
                // прочитать веса из файла
                case MemoryMode.GET:

                    tmpStrWeights = File.ReadAllLines(weightsFilePath);

                    for (int i = 0; i < neuronsCount; i++)
                    {
                        string[] memoryElement = tmpStrWeights[i].Split(delim);

                        for (int j = 1; j < prevNeuronsCount + 1; j++)
                        {
                            weights[i, j] = double.Parse(memoryElement[j - 1].Replace(',', '.'), 
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                // записать в файл веса
                case MemoryMode.SET:

                    tmpStrWeights = new string[neuronsCount];

                    for (int i = 0; i < neuronsCount; i++)
                    {
                        tmpStr = Neurons[i].Weights[0].ToString();
                        for (int j = 1; j < prevNeuronsCount + 1; j++)
                        {
                            tmpStr += delim[0] + Neurons[i].Weights[j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(weightsFilePath, tmpStrWeights);
                    
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

                    double[] tmpArr = new double[prevNeuronsCount + 1];
                    
                    tmpStrWeights = new string[neuronsCount];
                    tmpStr = "";

                    for (int i = 0; i < neuronsCount; i++)
                    {
                        for (int j = 0; j < prevNeuronsCount + 1; j++)
                        {
                            tmpArr[j] = 0.02 * random.NextDouble() - 0.01;
                        }

                        double tmpRatio = 1.0d / Math.Sqrt(Utils.Dispersion(tmpArr) * (prevNeuronsCount + 1));
                        double tmpShift = Utils.Average(tmpArr);

                        weights[i, 0] = (tmpArr[0] - tmpShift) * tmpRatio;
                        tmpStr = weights[i, 0].ToString();

                        for (int j = 1; j < prevNeuronsCount + 1; j++)
                        {
                            weights[i, j] = (tmpArr[j] - tmpShift) * tmpRatio;
                            tmpStr += delim[0] + weights[i, j].ToString();
                        }

                        tmpStrWeights[i] = tmpStr;
                    }

                    File.WriteAllLines(weightsFilePath, tmpStrWeights);
                    break;
            }

            return weights;
        }

        public abstract void Recognize(NeuroNet net, Layer nextLayer);
        public abstract double[] BackwardPass(double[] stuff);
    }
}

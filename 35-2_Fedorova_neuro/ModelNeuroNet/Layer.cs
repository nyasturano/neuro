using System;
using System.IO;
using System.Windows.Forms;


namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    abstract class Layer
    {
        protected string name_Layer; // наименование слоя

        private string pathDirWeights; // путь к каталогу весов
        private string pathFileWeights; // путь к файлу весов

        protected int numofneurons; // число нейронов в текущем слое
        protected int numofprevneurons; // число нейронов в предыдущем слое

        protected const double learningrate = 0.5;
        protected const double momentum = 0.05;
        protected double[,] lastdeltaweights;

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

        protected Layer(int non, int nonp, TypeNeuron nt, string nm_Layer)
        {
            name_Layer = nm_Layer;
            numofneurons = non;
            numofneurons = nonp;

            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";

            Neurons = new Neuron[non];

            double[,] Weights;

            if (File.Exists(pathFileWeights))
            {
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights);
            }
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            lastdeltaweights = new double[non, nonp + 1];

            for (int i = 0; i < non; i++)
            {
                double[] tmp_weights = new double[nonp + 1];
                for (int j = 0; j < nonp + 1; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt);
            }
        }

        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' }; // разделители слов
            string tmpStr; // временная строка для чтения
            string[] tmpStrWeights; // временный массив строк
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);
                    string[] memory_element;
                    for (int i = 0; i < numofneurons; i++)
                    {
                        memory_element = tmpStrWeights[i].Split(delim);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j].Replace(',', '.'), 
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                case MemoryMode.SET:

                    tmpStrWeights = new string[numofneurons];

                    if (!File.Exists(path))
                    {
                        MessageBox.Show("Файл " + name_Layer + " _memory.csv не найден. Создастся новый файл.", 
                            "Внимание", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                    }

                    for (int i = 0; i < numofneurons; i++)
                    {
                        tmpStr = Neurons[i].Weights[0].ToString();
                        for (int j = 1; j < numofprevneurons + 1; j++)
                        {
                            tmpStr += delim[0] + Neurons[i].Weights[j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    
                    break;


                case MemoryMode.INIT:

                    if (!File.Exists(path))
                    {
                        MessageBox.Show("Файл " + name_Layer + " _memory.csv не найден. Создастся новый файл.",
                            "Внимание", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                    }

                    /*
                        1. Веса инициализируются случайными величинами
                        2. Матожидание (среднее значение) всех весов нейрона должно равняться нулю
                         -> Найти среднее значение, его вычесть из всех значений весов
                        3. Среднеквадратическое отклонение случайных величин должно быть = 1
                     */

                    Random random = new Random();

                    tmpStrWeights = new string[numofneurons];
                    tmpStr = "";

                    for (int i = 0; i < numofneurons; i++)
                    {
                        // заполнение весов

                        for (int j = 1; j < numofprevneurons + 1; j++)
                        {
                            tmpStr += delim[0] + weights[i, j].ToString();
                        }
                        tmpStrWeights[i] = tmpStr;
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    break;
            }


            return weights;
        }

        public abstract void Recognize(NeuroNet net, Layer nextLayer);
        public abstract double[] BackwardPass(double[] stuff);
    }
}

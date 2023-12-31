﻿namespace _35_2_Fedorova_neuro.ModelNeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int neuronsCount, int prevNeuronsCount, NeuronType neuronType, string name) 
            : base(neuronsCount, prevNeuronsCount, neuronType, name) { }

        public override void Recognize(NeuroNet net, Layer nextLayer)
        {
            double[] hidden_out = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++) 
            {
                hidden_out[i] = Neurons[i].Output;
            }
            nextLayer.Data = hidden_out;
        }

        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[prevNeuronsCount];

            // вычисление градиетных сумм
            for (int j = 0; j < gr_sum.Length; j++)
            {
                double sum = 0;
                for (int k = 0; k < Neurons.Length; k++)
                {
                    sum += Neurons[k].Weights[j] * Neurons[k].Derivative * gr_sums[k];
                }
                gr_sum[j] = sum;
            }

            // вычисление коррекции синаптических весов
            for (int i = 0; i < neuronsCount; i++)
            {
                for (int n = 0; n < prevNeuronsCount + 1; n++)
                {
                    double delta_w = 0;
                    // для коррекции порога
                    if (n == 0)
                    {
                        delta_w = momentum * lastDeltaWeights[i, 0] + learningRate * Neurons[i].Derivative * gr_sums[i];
                    }
                    // коррекция синаптических весов
                    else
                    {
                        delta_w = momentum * lastDeltaWeights[i, n] + 
                            learningRate * Neurons[i].Inputs[n - 1] * Neurons[i].Derivative * gr_sums[i];
                    }

                    lastDeltaWeights[i, n] = delta_w;
                    Neurons[i].Weights[n] += delta_w; // коррекция весов
                }

            }
            return gr_sum;
        }
        
    }
}

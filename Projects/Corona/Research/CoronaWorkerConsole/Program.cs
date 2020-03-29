using CoronaWorkerBusinessLayer.WorkFlow;
using System;

namespace CoronaWorkerConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Data collection begin..");
            DataCollector.CollectCoronaDataFromWorldometers();
            Console.ReadKey();
        }
    }
}

using CoronaWorkerBusinessLayer.WorkFlow;
using CoronaWorkerLibrary;
using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using System.Reflection;

namespace CoronaWorkerConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Data collection begin..");

                var builder = new ConfigurationBuilder()
                    .SetBasePath(Directory.GetCurrentDirectory())
                    .AddJsonFile("AppSettings.json");

                var configuration = builder.Build();

                Initialization.CoronaDBSqlString = configuration["coronaDatabaseConnectionString"];
                
                DataCollector.CollectCoronaDataFromWorldometers();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }

            Console.ReadKey();
        }
    }
}

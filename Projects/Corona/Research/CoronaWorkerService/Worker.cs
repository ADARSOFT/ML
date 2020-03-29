using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using CoronaWorkerBusinessLayer.WorkFlow;
namespace CoronaWorkerService
{
    public class Worker : BackgroundService
    {
        private const int V = 21600000;
        private readonly ILogger<Worker> _logger;
        private static readonly long delayTimeProduction = V;

        public Worker(ILogger<Worker> logger)
        {
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                // _logger.LogInformation("Worker running at: {time}", DateTimeOffset.Now);

                DataCollector.CollectCoronaDataFromWorldometers();

                await Task.Delay(1000, stoppingToken);
            }
        }
    }
}

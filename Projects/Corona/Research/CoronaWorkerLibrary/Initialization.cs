using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaWorkerLibrary
{
    public class Initialization
    {
        public static readonly Dictionary<string, string> WebLocations = new Dictionary<string, string>()
        {
            { "Worldometers", "https://www.worldometers.info/coronavirus/" }
        };

        public static string CoronaDBSqlString = string.Empty;
    }
}

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

        public static readonly string CoronaDBSqlString = @"Server=DAMIR-ADARSOFT\MSSQLSERVER2014S;Initial Catalog=Corona;User Id=CoronaUser;Password=Corona123$!;";
    }
}

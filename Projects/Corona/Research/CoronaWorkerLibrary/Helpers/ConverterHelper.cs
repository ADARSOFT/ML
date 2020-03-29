using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaWorkerLibrary.Helpers
{
    public static class ConverterHelper
    {
        public static long ConvertStringToLong(string input)
        {
            input = input.Replace(",", "");
            input = input.Replace("+", "");

            long.TryParse(input, out long result);

            return result;
        }

        public static decimal ConvertStringToDecimal(string input)
        {
            decimal.TryParse(input, out decimal result);

            return result;
        }
    }
}

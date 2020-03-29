using System;
using System.Collections.Generic;
using System.Text;

namespace CoronaWorkerLibrary.Models
{
    public class ShortDetailsModel
    {
        public string CountryName { get; set; }
        public long TotalCasses { get; set; }
        public long NewCasses { get; set; }
        public long TotalDeaths { get; set; }
        public long NewDeaths { get; set; }
        public long TotalRecovered { get; set; }
        public long ActiveCasses { get; set; }
        public long SeriousCritical { get; set; }
        public decimal TopCases_1M_Population { get; set; }
        public decimal TopDeaths_1M_Population { get; set; }
        public string FirstCaseDate { get; set; }
    }
}

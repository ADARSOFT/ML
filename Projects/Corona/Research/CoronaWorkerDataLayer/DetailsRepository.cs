using CoronaWorkerLibrary;
using CoronaWorkerLibrary.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;

namespace CoronaWorkerDataLayer
{
    public class DetailsRepository
    {
        public void InsertMultipleDetails(List<ShortDetailsModel> details)
        {

            using (var connection = new SqlConnection(Initialization.CoronaDBSqlString))
            {
                
                using (var cmd = new SqlCommand("dbo.ShortDetailsInsertMultiple", connection))
                {
                    cmd.CommandType = CommandType.StoredProcedure;

                    cmd.Parameters.Add(BEChangesEmailDetailsList(details));
                    
                    connection.Open();
                    
                    cmd.ExecuteNonQuery();

                    connection.Close();
                }
            }
        }

        public static SqlParameter BEChangesEmailDetailsList(IList<ShortDetailsModel> request)
        {
            string udtName = "[dbo].[ShortDetailsByCountryList]";

            var dt = new DataTable(udtName);

            dt.Columns.Add("CountryName", typeof(string));
            dt.Columns.Add("TotalCasses", typeof(long));
            dt.Columns.Add("NewCasses", typeof(long));
            dt.Columns.Add("TotalDeaths", typeof(long));
            dt.Columns.Add("NewDeaths", typeof(long));
            dt.Columns.Add("TotalRecovered", typeof(long));
            dt.Columns.Add("ActiveCasses", typeof(long));
            dt.Columns.Add("SeriousCritical", typeof(long));
            dt.Columns.Add("TopCases_1M_Population", typeof(decimal));
            dt.Columns.Add("TopDeaths_1M_Population", typeof(decimal));
            dt.Columns.Add("FirstCaseDate", typeof(string));

            foreach (var shortDetails in request)
                dt.Rows.Add(new object[]
                    {
                        shortDetails.CountryName,
                        shortDetails.TotalCasses,
                        shortDetails.NewCasses,
                        shortDetails.TotalDeaths,
                        shortDetails.NewDeaths,
                        shortDetails.TotalRecovered,
                        shortDetails.ActiveCasses,
                        shortDetails.SeriousCritical,
                        shortDetails.TopCases_1M_Population,
                        shortDetails.TopDeaths_1M_Population,
                        shortDetails.FirstCaseDate
                    });

            var parameter = new SqlParameter();
            parameter.ParameterName = "@ShortDetailsByCountryList";
            parameter.TypeName = udtName;
            parameter.SqlDbType = SqlDbType.Structured;
            parameter.Value = dt;

            return parameter;
        }
    }
}

using CoronaWorkerDataLayer;
using CoronaWorkerLibrary;
using CoronaWorkerLibrary.Helpers;
using CoronaWorkerLibrary.Models;
using HtmlAgilityPack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;

namespace CoronaWorkerBusinessLayer.WorkFlow
{
    public class DataCollector
    {
        public static void CollectCoronaDataFromWorldometers()
        {
            Initialization.WebLocations.TryGetValue("Worldometers", out string uri);
            DetailsRepository detailsRepository = new DetailsRepository();

            try
            {
                WebClient webClient = new WebClient();

                string html = webClient.DownloadString(uri);

                var doc = new HtmlDocument();

                doc.LoadHtml(html);

                HtmlNode specificNode = doc.GetElementbyId("main_table_countries_today");

                HtmlNode tBody = specificNode.SelectSingleNode("tbody");

                List<ShortDetailsModel> shortDetailsList = new List<ShortDetailsModel>();

                foreach (var tr in tBody.ChildNodes.Where(p => p.Name == "tr"))
                {
                    int counter = 1;

                    ShortDetailsModel shortDetails = new ShortDetailsModel();

                    var tdNodes = tr.ChildNodes;

                    foreach (var td in tr.ChildNodes.Where(p => p.Name == "td"))
                    {
                        SetDetails(counter, shortDetails, td);

                        counter++;
                    }

                    shortDetailsList.Add(shortDetails);
                }

                detailsRepository.InsertMultipleDetails(shortDetailsList);
            }
            catch (Exception)
            {
                throw;
            }
        }

        private static void SetDetails(int counter, ShortDetailsModel shortDetails, HtmlNode td)
        {
            switch (counter)
            {
                case 1:
                    shortDetails.CountryName = td.InnerText;
                    break;
                case 2:
                    shortDetails.TotalCasses = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 3:
                    shortDetails.NewCasses = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 4:
                    shortDetails.TotalDeaths = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 5:
                    shortDetails.NewDeaths = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 6:
                    shortDetails.TotalRecovered = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 7:
                    shortDetails.ActiveCasses = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 8:
                    shortDetails.SeriousCritical = ConverterHelper.ConvertStringToLong(td.InnerText);
                    break;
                case 9:
                    shortDetails.TopCases_1M_Population = ConverterHelper.ConvertStringToDecimal(td.InnerText);
                    break;
                case 10:
                    shortDetails.TopDeaths_1M_Population = ConverterHelper.ConvertStringToDecimal(td.InnerText);
                    break;
                case 11:
                    shortDetails.FirstCaseDate = td.InnerText.Replace("\n","");
                    break;
                default:
                    break;
            }
        }
    }
}

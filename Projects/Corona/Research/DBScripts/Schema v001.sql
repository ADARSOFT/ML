IF DB_ID('Corona') IS NULL
BEGIN
	CREATE DATABASE Corona
END
ELSE
BEGIN
	PRINT('Database already exists..')
END
GO
USE Corona
GO

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='DownloadSession' and xtype='U')
BEGIN
	CREATE TABLE DownloadSession
	(
		SessionId BIGINT PRIMARY KEY IDENTITY,
		CreatedDate DATETIME
	)
END

GO
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ShortDetailsByCountry' and xtype='U')
BEGIN
	CREATE TABLE ShortDetailsByCountry
	(
		ShortDetailsId BIGINT IDENTITY PRIMARY KEY,
		CountryName NVARCHAR(50),
		TotalCasses BIGINT,
		NewCasses BIGINT,
		TotalDeaths BIGINT,
		NewDeaths BIGINT,
		TotalRecovered BIGINT,
		ActiveCasses BIGINT,
		SeriousCritical BIGINT,
		TopCases_1M_Population DECIMAL(17,2),
		TopDeaths_1M_Population DECIMAL(17,2),
		FirstCaseDate NVARCHAR(50) DEFAULT NULL,
		SessionId BIGINT,
		CONSTRAINT FK_DownloadSession_SessionId_ShortDetailsByCountry FOREIGN KEY (SessionId) REFERENCES DownloadSession(SessionId) 
	)
END
GO

IF TYPE_ID(N'ShortDetailsByCountryList') IS NULL
BEGIN
	CREATE TYPE ShortDetailsByCountryList AS TABLE 
	(
		CountryName NVARCHAR(50),
		TotalCasses BIGINT,
		NewCasses BIGINT,
		TotalDeaths BIGINT,
		NewDeaths BIGINT,
		TotalRecovered BIGINT,
		ActiveCasses BIGINT,
		SeriousCritical BIGINT,
		TopCases_1M_Population DECIMAL(17,2),
		TopDeaths_1M_Population DECIMAL(17,2),
		FirstCaseDate NVARCHAR(50)
	)
END
GO

IF EXISTS (SELECT * FROM sys.objects WHERE type = 'P' AND OBJECT_ID = OBJECT_ID('ShortDetailsInsertMultiple'))
BEGIN
	DROP PROCEDURE ShortDetailsInsertMultiple
END
GO

CREATE PROCEDURE ShortDetailsInsertMultiple
(
	@ShortDetailsByCountryList AS ShortDetailsByCountryList READONLY
)
AS
BEGIN
	INSERT INTO DownloadSession (CreatedDate) VALUES(GETDATE())
	DECLARE @DownloadSessionId BIGINT = SCOPE_IDENTITY()

	INSERT INTO ShortDetailsByCountry 
	(
		CountryName, 
		TotalCasses, 
		NewCasses, 
		TotalDeaths, 
		NewDeaths, 
		TotalRecovered, 
		ActiveCasses, 
		SeriousCritical, 
		TopCases_1M_Population,
		TopDeaths_1M_Population,
		FirstCaseDate,
		SessionId
	)
	SELECT 
		CountryName, 
		TotalCasses, 
		NewCasses, 
		TotalDeaths, 
		NewDeaths, 
		TotalRecovered, 
		ActiveCasses, 
		SeriousCritical, 
		TopCases_1M_Population,
		TopDeaths_1M_Population,
		FirstCaseDate,
		@DownloadSessionId
	FROM @ShortDetailsByCountryList
END
GO

--DROP PROCEDURE ShortDetailsInsertMultiple
--DROP TYPE ShortDetailsByContryList
--DROP TABLE ShortDetailsByContry

/*
SELECT * FROM DownloadSession
SELECT * FROM ShortDetailsByCountry AS SD
INNER JOIN DownloadSession AS DS ON SD.SessionId = DS.SessionId
*/

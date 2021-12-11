from datetime import datetime
import os


INSULIN_FILE = os.path.join(os.path.dirname(__file__), 'InsulinData.csv')
CGM_FILE = os.path.join(os.path.dirname(__file__), 'CGMData.csv')
RESULT_FILE = os.path.join(os.path.dirname(__file__), 'Results.csv')


def getMeanPercentageMetrics(in_data):
    """ Get mean of percentage value of metrics """

    hrgList = []
    hrgCrtList = []
    rngList = []
    rsList = []
    hog1List = []
    hog2List = []

    for date in in_data:
        hrgSum = 0
        hrgCrtSum = 0
        rngSum = 0
        rsSum = 0
        hog1Sum = 0
        hog2Sum = 0
        for reading in in_data[date].values():
            if reading > 180:
                hrgSum += 1
            if reading > 250:
                hrgCrtSum += 1
            if reading >= 70 and reading <= 180:
                rngSum += 1
            if reading >= 70 and reading <= 150:
                rsSum += 1
            if reading < 70:
                hog1Sum += 1
            if reading < 54:
                hog2Sum += 1
        # for getting %
        hrgList.append(hrgSum/2.88)
        hrgCrtList.append(hrgCrtSum/2.88)
        rngList.append(rngSum/2.88)
        rsList.append(rsSum/2.88)
        hog1List.append(hog1Sum/2.88)
        hog2List.append(hog2Sum/2.88)

    print(len(in_data))

    avgList = getMeanList(
        hrgList, 
        hrgCrtList, 
        rngList, 
        rsList,
        hog1List,
        hog2List
    )

    return avgList


def getMeanList(
        hrgList, 
        hrgCrtList, 
        rngList, 
        rsList,
        hog1List,
        hog2List
    ):
    """ Calculate mean values from given lists """

    avgList = [0,0,0,0,0,0]

    for hrg in hrgList:
        avgList[0] += hrg
    avgList[0] /= len(hrgList)

    for hrgCrt in hrgCrtList:
        avgList[1] += hrgCrt
    avgList[1] /= len(hrgCrtList)

    for rng in rngList:
        avgList[2] += rng
    avgList[2] /= len(rngList)

    for rs in rsList:
        avgList[3] += rs
    avgList[3] /= len(rsList)

    for hog1 in hog1List:
        avgList[4] += hog1
    avgList[4] /= len(hog1List)

    for hog2 in hog2List:
        avgList[5] += hog2
    avgList[5] /= len(hog2List)
    
    return avgList


def convertToString(in_list):
    """ Convert list into CSV string """

    res = ''
    for elem in in_list:
        res += "{},".format(str(elem))
    return res


def getAutoDate():
    """ Get date when AUTO is enabled """

    autoDate = ''
    with open(INSULIN_FILE, 'r') as f:    
        i = 0
        for line in f:
            i += 1
            if i == 1:
                continue
                
            arr = line.split(",")        
            if arr[16] == 'AUTO MODE ACTIVE PLGM OFF':
                autoDate = arr[1]
    
    print()
    return autoDate


def getTimeReadingDict():
    """ Retrieve data from input file in the form - {timestamp: reading} """

    timeReadingDict = {}
    with open(CGM_FILE, 'r') as f:
        i = 0
        for line in f:
            i += 1
            if i == 1:
                continue
                
            arr = line.split(",")
            dateTime = "{0}-{1}".format(arr[1], arr[2])
            reading = 0
            if arr[30] == '':
                # reading = 0
                continue
            else:
                reading = int(arr[30])
                
            timeReadingDict[dateTime] = reading

    # print(len(timeReadingDict.keys()))
    return timeReadingDict


def getDateReadingListDict(timeReadingDict):
    """ Aggregate data on the basis of date """

    # print(len(timeReadingDict.keys()))
    dateReadingListDict = {}
    distinct = set()

    for time in timeReadingDict:
        date, clock = time.split('-')
        distinct.add(date)
        if date not in dateReadingListDict:
            dateReadingListDict[date] = {}
        dateReadingListDict[date][clock] = timeReadingDict[time]

    # print(dateReadingListDict.keys())
    # print(distinct)

    return dateReadingListDict


def writeResult(
        manNightAvgList,
        manDayAvgList,
        manWholeAvgList,
        autoNightAvgList,
        autoDayAvgList,
        autoWholeAvgList
    ):
    """ Write the results into Results.csv file """

    with open(RESULT_FILE, 'w') as f:
        f.write(convertToString(manNightAvgList+manDayAvgList+manWholeAvgList))
        f.write('1.1\n')
        f.write(convertToString(autoNightAvgList+autoDayAvgList+autoWholeAvgList))
        f.write('1.1')


def driver():
    """ Extract metric information from file """

    autoDate = getAutoDate()    
    timeReadingDict = getTimeReadingDict()
    dateReadingListDict = getDateReadingListDict(timeReadingDict)
    # print(len(dateReadingListDict.keys()))

    autoDatesReadingListDict = {}
    manDatesReadingListDict = {}
    switchDate = datetime.strptime(autoDate, "%m/%d/%Y")
    for date in dateReadingListDict:
        curr = datetime.strptime(date, "%m/%d/%Y")
        if curr >= switchDate:
            autoDatesReadingListDict[date] = dateReadingListDict[date]
        if curr <= switchDate:
            manDatesReadingListDict[date] = dateReadingListDict[date]
    
    # print(len(autoDatesReadingListDict))
    # print(len(manDatesReadingListDict))

    autoDayDict = {}
    autoNightDict = {}
    manDayDict = {}
    manNightDict = {}
    dayLimit = datetime.strptime("06:00:00", "%H:%M:%S")
    for date in autoDatesReadingListDict:
        for time in autoDatesReadingListDict[date]:
            curr = datetime.strptime(time, "%H:%M:%S")
            if curr < dayLimit:
                if date not in autoNightDict:
                    autoNightDict[date] = {}
                autoNightDict[date][time] = autoDatesReadingListDict[date][time]
            else:
                if date not in autoDayDict:
                    autoDayDict[date] = {}
                autoDayDict[date][time] = autoDatesReadingListDict[date][time]
                
    for date in manDatesReadingListDict:
        for time in manDatesReadingListDict[date]:
            curr = datetime.strptime(time, "%H:%M:%S")
            if curr < dayLimit:
                if date not in manNightDict:
                    manNightDict[date] = {}
                manNightDict[date][time] = manDatesReadingListDict[date][time]
            else:
                if date not in manDayDict:
                    manDayDict[date] = {}
                manDayDict[date][time] = manDatesReadingListDict[date][time]

    autoWholeAvgList = getMeanPercentageMetrics(autoDatesReadingListDict)
    autoDayAvgList = getMeanPercentageMetrics(autoDayDict)
    autoNightAvgList = getMeanPercentageMetrics(autoNightDict)
    manWholeAvgList = getMeanPercentageMetrics(manDatesReadingListDict)
    manDayAvgList = getMeanPercentageMetrics(manDayDict)
    manNightAvgList = getMeanPercentageMetrics(manNightDict)

    writeResult(
        manNightAvgList,
        manDayAvgList,
        manWholeAvgList,
        autoNightAvgList,
        autoDayAvgList,
        autoWholeAvgList
    )


if __name__ == "__main__":
    driver()
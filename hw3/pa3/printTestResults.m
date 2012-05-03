function printTestResults(nrmError, maxAllowedError, testName)
	if nrmError <= maxAllowedError
		successStr = 'PASSED';
	else
		successStr = 'FAILED';
	end
	fprintf ('\n%s\n',testName);
	fprintf ('\nNorm error : %f \nMaximum allowed error : %f \n', nrmError, maxAllowedError);
	fprintf ('%s %s\n',testName, successStr);
end

package edu.autovolweb;

import java.util.ArrayList;
import java.util.List;

import javax.servlet.annotation.WebServlet;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Servlet implementation class AvgRfClassifyServlet
 */
@WebServlet("/AvgRfClassifyServlet")
public class AvgRfClassifyServlet extends RfClassifyServlet {
	private static final long serialVersionUID = 1L;

    @Override
	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		Instances avgData = AvgKnnClassifyServlet.avgData(data, 8);
		return avgData;
	}
    
	@Override
	protected Instance constructTarget(String input, String userId) {
		List<CurrentStateData> states = CurrentStateUtil.fromJson(input);
		
		List<Instance> instances = new ArrayList<Instance>();
		for (CurrentStateData state : states) {
			Instance target = CurrentStateUtil.toInstance(state);
			instances.add(target);
		}
		
		Instance avg = DataUploadServlet.avgInstances(instances, instances.get(0).dataset());
		return avg;
	}

}

package edu.autovolweb;


import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import edu.autovolweb.CurrentStateData;
import edu.autovolweb.CurrentStateUtil;
import edu.autovolweb.ViewDataServlet;

/**
 * Servlet implementation class KnnClassifyServlet
 */
@WebServlet("/KnnClassifyServlet")
public class KnnClassifyServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	
	private Map<String, KDTree> kdMap;
	private Set<String> initializedUsers;
	
	private static int[] ks = new int[] {3, 7, 13};

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String userId = request.getParameter("user");
		if (userId == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
		if (!initializedUsers.contains(userId)) {
			try {
				initForUser(userId);
			} catch (Exception e) {
				response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
				return;
			}
			
		}
		initializedUsers.add(userId);
		String input = request.getParameter("target");
		if (input == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
		
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);
		Instance target = CurrentStateUtil.toInstance(state);
		JsonObject json = new JsonObject();
		KDTree kd = kdMap.get(userId);
		for (int k : ks) {
			String label = kdClassify(target.classAttribute(), kd, target, k);
			json.addProperty("" + k, label);
		}
		
		response.getWriter().write(json.toString());
	}
	
	@Override
	public void init() throws ServletException {
		kdMap = new HashMap<>();
		initializedUsers = Collections.synchronizedSet(new HashSet<String>());
	}
	
	private void initForUser(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		data.setClass(data.attribute("ringer"));
		KDTree kd = new KDTree();
		kdMap.put(userId, kd);
		
		kd.setInstances(data);
	}
	
    private String kdClassify(Attribute classAttr, KDTree kd, Instance target, int k) {
    	String result = "err";
		try {
			Instances neighbors = kd.kNearestNeighbours(target, k);
			int numClassValues = classAttr.numValues();
			int[] classCounts = new int[numClassValues];
			for (Instance n : neighbors) {
				classCounts[(int) Math.round(n.classValue())]++;
			}
			
			int maxIndex = 0;
			int maxCount = Integer.MIN_VALUE;
			for (int i = 0; i < numClassValues; i++) {
				if (classCounts[i] > maxCount) {
					maxCount = classCounts[i];
					maxIndex = i;
				}
			}
			
			return classAttr.value(maxIndex);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
    	return result;
    }

}

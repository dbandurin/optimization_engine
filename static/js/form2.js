var referenceId = account_id + '_Strike_' + strike_id

// Get Summary Table
function getSummaryData(account_id, strike_id) {
  var summaryDataUrl = 'static/Account_' + referenceId + '/Results/json/summary.json';
  $('#summaryTable').dataTable({
    "bDestroy":true,
    "sAjaxSource": summaryDataUrl,
    "fnServerData": function (sSource, aoData, fnCallback ) {
      var returnData = [];
      for (var i = 0; i < aoData.length; i++) {
        returnData.push($.makeArray(aoData[i]));
      }
      $.getJSON( sSource, returnData , function (json) { 
        fnCallback(json)
      } );
    },
    "paging":   false,
    "ordering": false,
    "info":     false,
    "bFilter": false,

    "columnDefs": [
        {
        data: "budget_per_day",
        targets: 0,
        render: $.fn.dataTable.render.number( ',', '.', 2, '' )
      },
      {
        data: "media_spend_today",
        targets: 1,
        render: $.fn.dataTable.render.number( ',', '.', 2, '' )
      },
      {
        data: "media_spend_total",
        targets: 2,
        render: $.fn.dataTable.render.number( ',', '.', 2, '' )
      },
      {
        data: "budget_frac_diff_today",
        targets: 3
      },
      {
        data: "budget_frac_diff_yesterday",
        targets: 4
      },
      {
        data: "media_order",
        targets: 5,
        render: $.fn.dataTable.render.number( ',', '.', 2, '' )
      },
      {
        data: "max_cpv",
        targets: 6
      },
      {
        data: "cpv_so_far",
      targets: 7
      },
      {
        data: "vr_so_far",
      targets: 8
      },
      {
        data: "view_order",
      targets: 9
      },
      {
        data: "views_so_far",
      targets: 10
      },
      {
        data: "days_left",
      targets: 11
      },
      {
        data: "days_passed",
      targets: 12
      },
      {
        data: "last_update",
        targets: 13
      },
      {
        data: "paused_frac",
        targets: 14
      }
    ],
    "initComplete": function(settings, json) {
      $("#summary .panel").show();
    }
  }); 
}


// Get Optimized View data
function getOptimizedData(account_id, strike_id) {
  var optimizedDataUrl = 'static/Account_' + referenceId + '/Results/json/opt_res.json';
  $('#optimization_view_table').dataTable({
    "bDestroy":true,
    "sAjaxSource": optimizedDataUrl,
    "fnServerData": function (sSource, aoData, fnCallback ) {
      var returnData = [];
      for (var i = 0; i < aoData.length; i++) {
        returnData.push($.makeArray(aoData[i]));
      }
      $.getJSON( sSource, returnData , function (json) { 
        fnCallback(json)
      } );
    },
    "paging":   false,
    "ordering": false,
    "info":     false,
    "bFilter": false,
    
    "columnDefs": [
      {
        data: "Budget",
        targets: 0,
        render: $.fn.dataTable.render.number( ',', '.', 2, '' ) 
      },
      {
        data: "Views",
        targets: 1
      },
      {
        data: "ViewRate",
        targets: 2
      },
      {
        data: "ViewCompletionRate",
        targets: 3
      },
      {
        data: "CTR",
        targets: 4
      },
      {
        data: "CPV",
        targets: 5
      },
      // {
      //   data: "Cost_ratio", 
      //   targets: 5
      // },
      {
        data: "EffViews_ratio", 
        targets: 6
      },
      //{
      //  data: "MaxBudgetIncrease",
      //  targets: 7
      //},
      {
        data: "VR_importance",
        targets: 7
      }      
    ],
    "initComplete": function(settings, json) {
      $("#optimization_result .panel").show();
    }
  }); 
}


// Get Optimized Budget Table
function getBudgetData(account_id, strike_id) {
  var budgetDataUrl = 'static/Account_' + referenceId + '/Results/json/budget_opt.json';
  $.get(budgetDataUrl)
    .done(function() { 
      $('#budgetTable').dataTable({
        "bDestroy":true,
        "sAjaxSource": budgetDataUrl,
        "hideEmptyCols": true,
        "oTable.fnSort": [[23, "desc" ]],
        "fnServerData": function (sSource, aoData, fnCallback ) {
          var returnData = [];
          for (var i = 0; i < aoData.length; i++) {
            returnData.push($.makeArray(aoData[i]));
          }
          $.getJSON( sSource, returnData , function (json) { 
            fnCallback(json)
          } );
        },
        "dom": '<"top"if>rt<"bottom"lp><"clear">',
        "columnDefs": [
          {
            class: 'first-column',
            data: "campaign",
            targets: 0,
            width: "20%"
          },
          {
            data: "campaign_id",
            targets: 1
          },
          {
            data: "campaign_state",
            targets: 2
          },
          {
            data: "network",
            targets: 3
          },
          {
            data: "placement",
            targets: 4
          }, 
          {
            data: "video_id",
            targets: 5
          },                     
          {
            data: "device",
            targets: 6
          },
          {
            data: "age_range",
            targets: 7
          },  
          {
            data: "gender",
            targets: 8
          }, 
          {
            data: "impressions",
            targets: 9
          },   
          {
            data: "views",
            targets: 10
          },
          {
            data: "views_full",
            targets: 11
          },
          {
            data: "clicks",
            targets: 12
          },
          {
            data: "VR",
            targets: 13
          },
          {
            data: "VCR",
            targets: 14
          },
          {
            data: "CTR",
            targets: 15
          },
          {
            data: "CPV",
            targets: 16
          },
          {
            data: "cost",
            targets: 17,
            render: $.fn.dataTable.render.number( ',', '.', 2, '' ) 
          },
          {
            data: "Budget_max",
            targets: 18,
            render: $.fn.dataTable.render.number( ',', '.', 2, '' ) 
          },
          {
            data: "Score_to_cost",
            targets: 19,
            sClass: "highlight-blue"
          },
          {
            data: "Score",
            targets: 20,
            sClass: "highlight-blue"
          },
          {
            data: "Budget",
            targets: 21,
            render: $.fn.dataTable.render.number( ',', '.', 2, '' ),
            sClass: "highlight-blue"
          },
          {
            data: "CPV_bid",
            targets: 22,
            sClass: "highlight-blue"
          },
          {
            data: "CPV_max",
            targets: 23, 
            //"bVisible": false,
            //sClass: "highlight-blue hide"
            sClass: "highlight-blue"
          }
        ],
        
        "initComplete": function(settings, json) {
          $("#budget .panel").show();
        }
      }); 
    }); 
  }

function createBudgetPlot() {
  //console.log(plots)
  //var src = 'static/Account_' + referenceId + '/Results/plots/opt_views_vs_budget.png';
  //var plot = plots.replace(/\[|]|&#39;/g,'').split(',');
  //var src = plot
  //$('#plots .budget img').attr('src', src);
  var plot = plots.replace(/\[|]|&#39;/g,'').split(',');
  $.each(plot , function(i, val) { 
    src = plot[i].trim();
	console.log(src);
    // $('#plots .subplots').append('<img class="col-md-6" src="static/Account_' + referenceId + '/Results/plots/control_plots/' + src + '" />')
    $('#charts .main_plots').append('<img class="col-md-5" src=' + src + '>');
  });
}
function createCampaignStatus() {
  var src = 'static/Account_' + referenceId + '/Results/plots/All_campaigns_status.png';
  $('#charts .control_plots').prepend('<img class="col-md-6" src=' + src + ' />');
}
function createCostViewStatus() {
  var src = 'static/Account_' + referenceId + '/Results/plots/cost_and_views_status.png';
  $('#charts .control_plots').prepend('<img class="col-md-6" src=' + src + ' />');
}

function createControlPlots() {
  //console.log(control_plots)
  var plot = control_plots.replace(/\[|]|&#39;/g,'').split(',');
  $.each(plot , function(i, val) { 
    src = plot[i].trim();
	console.log(src);
    // $('#plots .subplots').append('<img class="col-md-6" src="static/Account_' + referenceId + '/Results/plots/control_plots/' + src + '" />')
    $('#charts .control_plots').append('<img class="col-md-6" src=' + src + '>');
  });
}

function waitForElement(){
  if(flag == 1){
    console.log('Loaded');
    waitingDialog.hide();
    getSummaryData(account_id, strike_id);
    getBudgetData(account_id, strike_id);
    getOptimizedData(account_id, strike_id);
    createBudgetPlot();
    //if(make_control_plots === "True") {
    if(make_control_plots != "None") {  
      createControlPlots();
      //createCampaignStatus();
      //createCostViewStatus();
    }
    flag = 0;
  }
  else{
    setTimeout(function(){
      console.log('Loading...');
      waitingDialog.show();
      waitForElement();
    },10000);
  }
}

function savePlan() {
  var saveForm = $("#putToDB");
  saveForm.submit(function(e){
    e.preventDefault();
    var $form = $(this);
    $.ajax({
     type     : "POST",
     cache    : false,
     url      : $form.attr('action'),
     data: $form.serialize(),
     // contentType: "application/json; charset=utf-8",
     datatype: 'json',
     success  : function(data) {
      result = JSON.parse(data);
      console.log(result);
      $form.find(".alert").remove();
      $form.append('<div class="alert alert-success" role="alert">' 
        + '<ul>' 
        + '<li>' + result.data[0].message + '</li>'
        + '<li> Update time: ' + result.data[0].cpv_update_time + '</li>'
        + '</ul>' 
        + '</div>')
     },
     error: function(data) {
       $form.append('<div class="alert alert-danger" role="alert">Not able to store data to DB. Please reload.</div>')
     }
    });
 })
}

function retrievePlan() {
  var saveForm = $("#getFromDB");
  saveForm.submit(function(e){
    e.preventDefault();
    var $form = $(this);
    $.ajax({
     type     : "POST",
     cache    : false,
     url      : $form.attr('action'),
     data: $form.serialize(),
     datatype: 'json',
     success  : function(data) {
       result = JSON.parse(data);
       console.log(result);
       $form.find(".download").remove();
       $('#get_plan_from_db').after("<a class='download btn btn-link' href=" + result.data[0].link + ">Download Budget " + result.data[0].version + "</a>")
     },
     error: function(data) {
       $formH.append('<div class="alert alert-danger" role="alert">Not able to save data to DB</div>')
     }
    });
 })
}


$(function() {
  waitForElement()
  $('#submit').click(function(e) {
    waitingDialog.show();
    waitForElement(account_id, strike_id);
  });

  $('#save_plan_to_db').click(function(e) {
    savePlan();
  });

  $('#getFromDB').click(function(e) {
    retrievePlan();
  });

});






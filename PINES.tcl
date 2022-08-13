#PINES.tcl contains tcltk procedures for conducting the PINES observing program on the Perkins
#The procedures read text files located in the current data directory. 

# proc PINES_set_date {set_date} {
#     global pines_date
#     set pines_date $set_date
#     puts [concat "PINES date (YYYYMMDD in UT) set to" $set_date]
# }

proc PINES_setMimir {} {

    puts "PINES_mimirset configures Mimir for PINES observations."

    #puts "Moving slit out of the field..."
    #slit pos=100
    #slit -home

    #puts "Moving decker out of the field..."
    #decker pos=100
    #decker -home

    #puts "Setting up F/5 camera..."
    #Cam_Home
    #Cam_F5
    
    puts "Switching to J band..."
    set_J

    puts "PINES_mimirset COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }


}

proc PINES_repoint {xnow ynow} {

    puts "Moving object at (xnow,ynow) to (700,382)"
    recenter $xnow $ynow 700 382 0.579

    puts "Taking 1 second test exposure..."
    test etime=1 nexp=1 {comment1=} {comment2=}

    puts "PINES_repoint COMPLETE."
    puts "If target is now near (700,382), update pointing with UC then U (MOVE)."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

}

proc PINES_mov2cor {xnow ynow} {

    puts "Moving object at (xnow,ynow) to (100,924)"
    recenter $xnow $ynow 100 924 0.579

    puts "Taking 1 second test exposure..."
    test etime=1 nexp=1 {comment1=} {comment2=}

    puts "PINES_move2corner COMPLETE."
    puts "Examine object elongation and adjust focus accordingly."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

}

proc PINES_domeflat {expt nexp} {
    
    #turn logging off
    Logging_OFF
    
    
    #turn lights on (should already be on to determine exp times)
    puts "Turning dome flat lamps on..."
    domeflat_on

    puts "Waiting 10 seconds for lamp to warm up..."
    after 10000

    set_J
    puts "Acquiring J-band flats lights on..."	    
    go etime=$expt title=dome_lamp_on nexp=$nexp {comment1=} {comment2=}
    
    #turn lights off
    domeflat_off

    puts "Waiting 10 seconds for lamp to cool down..."
    after 10000
    
    puts "Acquiring J flats lights off..."	    
    go etime=$expt title=dome_lamp_off nexp=$nexp {comment1=} {comment2=}
    

    #take a test to reset the comments
    puts "Taking a test exposure to reset comments..."	    
    test etime=0 nexp=1 title= {comment1=} {comment2=}

    puts "PINES_domeflat COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }


}

proc PINES_JHK_domeflat {J_expt H_expt K_expt nexp} {
    
    #turn logging off
    Logging_OFF
    
    for {set i 0} {$i < 3} {incr i} {
    
    #turn lights on (should already be on to determine exp times)
    puts "Turning dome flat lamps on..."
    domeflat_on

    puts "Waiting 10 seconds for lamp to warm up..."
    after 10000

    if {$i == 0} {
	set_J
	set expt $J_expt
	puts "Acquiring J-band flats lights on..."	    
    }
    if {$i == 1} {
	set_H
	set expt $H_expt
	puts "Acquiring H-band flats lights on..."	    
    }
    if {$i == 2} {
	set_K
	set expt $K_expt
	puts "Acquiring K-band flats lights on..."	    
    }
    
    go etime=$expt title=dome_lamp_on nexp=$nexp {comment1=} {comment2=}
    
    #turn lights off
    domeflat_off

    puts "Flats lamp off."	    

    puts "Waiting 10 seconds for lamp to cool down..."
    after 10000
    
    puts "Acquiring lamps off exposures..."
    go etime=$expt title=dome_lamp_off nexp=$nexp {comment1=} {comment2=}    

    }

    #take a test to reset the nexp and comments
    puts "Taking a test exposure to reset comments..."	    
    test etime=0 nexp=1 title= {comment1=} {comment2=}

    puts "PINES_JHK_domeflat COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }


}


proc PINES_linearity {} {
  
  puts "Running PINES_linearity: 1-15 seconds, 1 second intervals, 20 exposures with lamps on, 20 lamps off..."

  PINES_domeflat 1.0 20
  PINES_domeflat 2.0 20
  PINES_domeflat 3.0 20
  PINES_domeflat 4.0 20
  PINES_domeflat 5.0 20
  PINES_domeflat 6.0 20
  PINES_domeflat 7.0 20
  PINES_domeflat 8.0 20
  PINES_domeflat 9.0 20
  PINES_domeflat 10.0 20
  PINES_domeflat 11.0 20
  PINES_domeflat 12.0 20
  PINES_domeflat 13.0 20
  PINES_domeflat 14.0 20
  PINES_domeflat 15.0 20

}

proc PINES_darks {expt1 expt2 expt3 expt4 expt5 expt6 nexp} {

    set_Dark

    if {$expt1 > 0} {
	puts "Acquiring darks..."	    
	go etime=$expt1 title=dark nexp=$nexp {comment1=} {comment2=}
    }

    if {$expt2 > 0} {
	puts "Acquiring darks..."	    
	go etime=$expt2 title=dark nexp=$nexp {comment1=} {comment2=}
    }

    if {$expt3 > 0} {
	puts "Acquiring darks..."	    
	go etime=$expt3 title=dark nexp=$nexp {comment1=} {comment2=}
    }

    if {$expt4 > 0} {
	puts "Acquiring darks..."	    
	go etime=$expt4 title=dark nexp=$nexp {comment1=} {comment2=}
    }

    if {$expt5 > 0} {
	puts "Acquiring darks..."	    
	go etime=$expt5 title=dark nexp=$nexp {comment1=} {comment2=}
    }

    if {$expt6 > 0} {
	puts "Acquiring darks..."	    
	go etime=$expt6 title=dark nexp=$nexp {comment1=} {comment2=}
    }


    puts "Returning to J-band filter..."
    set_J 

    #take a test to reset the nexp and comments
    puts "Taking a test exposure to reset comments..."	    
    test etime=0 nexp=1 title= {comment1=} {comment2=}

    puts "PINES_darks complete."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

}


#Replaces PINES_set_date
proc PINES_get_date {} {

    #gets date in UT YYYYMMDD for writing reading files in the data directory
    
    #Add 10 hours to system time to ensure UT at sunset and sunrise
    set systemTime [expr [clock seconds] + 12 * 3600]
    set yr [clock format $systemTime -format %Y]
    set mn [clock format $systemTime -format %m]
    set day [clock format $systemTime -format %d]
    #return 20210728
    return $yr$mn$day
    
}

proc PINES_watchdog_get {start_file_time} {

    set pines_date [PINES_get_date]
    
    puts "Waiting until PINES_watchdog calculates shift/fwhm (15 sec timeout)..."
    set start_wait_time [ clock seconds ]
    set current_wait_time [ clock seconds ]
    set current_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]
    while { $current_wait_time < $start_wait_time + 15 && $current_file_time == $start_file_time } {
	#Pause for 0.1 second within forloop to slow it down
	after 100
	set current_wait_time [ clock seconds ]
	set current_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]
    }

    #Wait 0.5 seconds for file to be completely written
    after 500
    
    puts "Reading in text written by python program..."
    set fid [open /mimir/data/obs72/$pines_date/image_shift.txt r]

    #Read first line of image_shift.txt
    #set text [gets $fid]
    gets $fid text
    set list [split $text " "]
    
    set d_ra [expr [lindex $list 0]]
    set d_dec [expr [lindex $list 1]]
    set name [lindex $list 2]
    set fwhm [lindex $list 3]

    #Read the second line of image_shift.txt if it exists
    #set text2 [gets $fid]
    gets $fid text2
    if { $text2 == ""
    } then { set d_ra_median 0.0
	set d_dec_median 0.0
    } else { set list2 [split $text2 " "]    
	set d_ra_median [expr [lindex $list2 0]]
	set d_dec_median [expr [lindex $list2 1]]
    }

    close $fid
    
    return [list $d_ra $d_dec $name $fwhm $d_ra_median $d_dec_median]

}

proc PINES_focus {expt current_focus} {

    puts "STARTING PINES_focus..."

    set pines_date [PINES_get_date]

    #create list with focus positions
    set rel_focus_list { -200 -150 -100 -50 0 50 100 150 200}

    #create index
    set index 1

    #move focus inside start point, so we are always moving it out
    focus_go [expr $current_focus + [lindex $rel_focus_list 0] - 50]
    
    #iterate through focus positions
    foreach rel_focus $rel_focus_list {
	focus_go [expr $current_focus + $rel_focus]
	    
	#put zeros into old file
	exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

	#save time the file was modified
	set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

	puts "Taking a test exposure..."	    
	test etime=$expt nexp=1 {comment1=} {comment2=}


	#read output from PINES_watchdog
	set data [PINES_watchdog_get $start_file_time]
	set d_ra [expr [lindex $data 0]]
	set d_dec [expr [lindex $data 1]]
	set target_name [lindex $data 2]
	set fwhm [lindex $data 3]

	lappend fwhm_list $fwhm

	puts [concat "Focus=" [expr $current_focus + $rel_focus] ", FWHM=" $fwhm]

	#increment index
	incr index
    }
    
    #print out focus locations and fwhm
    set index 0
    puts "Focus FWHM:"
    foreach rel_focus $rel_focus_list {
	puts [concat [expr $current_focus + $rel_focus] [lindex $fwhm_list $index]]
	#increment index
	incr index
    }
    
    #find minimum fwhm
    set fwhm_sorted [lsort $fwhm_list]

    set best_index [lsearch $fwhm_list [lindex $fwhm_sorted 0]]
    set best_rel_focus [lindex $rel_focus_list $best_index]

    set index 1
    while {[lindex $fwhm_list $best_index] == -1} {
	set best_index [lsearch $fwhm_list [lindex $fwhm_sorted $index]]
	set best_rel_focus [lindex $rel_focus_list $best_index]
	incr index
	if {$index == 4} {break}
    }


    #move to best focus
    puts [concat "Moving to best focus of" [expr $current_focus + $best_rel_focus] "..."]
    focus_go [expr $current_focus + $best_rel_focus - 50]
    focus_go [expr $current_focus + $best_rel_focus]

    puts "Taking a final exposure..."	    
    test etime=$expt nexp=1 {comment1=} {comment2=}

    puts "PINES_focus COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }


}


proc PINES_peakup {expt} {

    set pines_date [PINES_get_date]

    puts "STARTING PINES_peakup..."

    set prop 1.0
    set integ 0.1
    set integrated_ra_error 0.0
    set integrated_dec_error 0.0

    #set maximum move for peak up in arcseconds
    set limit 300

    #take exposure and move tele 4 times, exit for loop is separation is good
    for {set i 0} {$i < 5} {incr i} {

	#put zeros into old file
	exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

	#save time the file was modified
	set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

	puts "Taking a test exposure..."	    
	test etime=$expt nexp=1 {comment1=} {comment2=}

	#read output from PINES_watchdog
	set data [PINES_watchdog_get $start_file_time]
	set d_ra [expr [lindex $data 0]]
	set d_dec [expr [lindex $data 1]]
	set target_name [lindex $data 2]
	set fwhm [lindex $data 3]

	puts [concat "Target name is" $target_name "."]

	set separation [expr sqrt(pow($d_ra,2) + pow($d_dec,2))]
    
	if {$separation == 0 
	} then { puts "PINES_watchdog not able to calculate shift, returned 0 0."
	} elseif {$separation < 2.0
	} then { puts "PINES_peakup successful, target within 2 arcseconds of master position."
	    set success 1
	    break
	} else {puts [concat "Target" $separation "arcseconds away from master position."]}

	if {$i == 4
	} then { puts [concat "PINES_peakup unsuccessful, target" $separation "arcseconds away from master position."]
	    set success -1
	    break
	} 

	#Move if the suggested motion is good

	if {[expr abs($d_ra)] == 0 && [expr abs($d_dec)] == 0
	} then {puts "Suggested RA and DEC moves equal to 0. PINES_watchdog likely failed. Move not executed."
	} elseif { [expr abs($d_ra)] < $limit && [expr abs($d_dec)] < $limit 
	       } then {
    	    if {$i > 0} then { set integrated_ra_error [expr $integrated_ra_error + $d_ra]
		set integrated_dec_error [expr $integrated_dec_error + $d_dec]
	    }	    

	    puts [concat "Moving "  [expr $prop * $d_ra + $integ * $integrated_ra_error] "," [expr $prop * $d_dec + $integ * $integrated_dec_error] " arcsec..."]
	    rmove d_ra=[expr $prop * $d_ra + $integ * $integrated_ra_error] d_dec=[expr $prop * $d_dec + $integ * $integrated_dec_error]

	    #puts [concat "Moving telescope" [expr $d_ra * $prop] "arcsec in RA and" [expr $d_ra * $prop] "in DEC..."]
	    #rmove d_ra=[expr $d_ra * $prop] d_dec=[expr $d_dec * $prop]	
	} else {puts [concat "Suggested RA or DEC move greater than" $limit "arcsec limit. Move not executed."]}
    }

    
    puts "PINES_peakup COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

    return $success
    
}

proc PINES_guide {expt total_time} {

    puts "STARTING PINES_guide..."

    set pines_date [PINES_get_date]

    # set target name to be whatever is in image_shift currently
    set fid [open /mimir/data/obs72/$pines_date/image_shift.txt r]
    set text [gets $fid]
    close $fid
    set list [split $text " "]
    set target_name [lindex $list 2]
 
    # #put zeros into old file
    # exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

    # #save time the file was modified
    # set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

    # puts "Taking a test exposure to get target name..."	    
    # test etime=1 nexp=1 {comment1=} {comment2=}

    # #read output from PINES_watchdog
    # set data [PINES_watchdog_get $start_file_time]
    # set d_ra [expr [lindex $data 0]]
    # set d_dec [expr [lindex $data 1]]
    # set target_name [lindex $data 2]
    # set fwhm [lindex $data 3]

    puts [concat "Target name is" $target_name "."]

    #set limits for guiding corrections in arcseconds
    set upper_limit 10.0
    set lower_limit 0.5

    #set guiding coefficients
    set prop 1.0
    set integ 0.1
    set integrated_ra_error 0.0
    set integrated_dec_error 0.0
    
    #get start time
    set start_time [ clock seconds ]

    puts [concat "Taking" $expt "second exposures for" $total_time "seconds..."]

    #set index to 1
    set index 1

    #get current time for while loop
    set current_time [ clock seconds ]

    #turn off display, telescope updates and logging to save time
    display_off
    set tmsg [ catch { Tele_Gui_OFF } ]
    Logging_OFF

    #start while loop
    while {$current_time < $start_time + $total_time - $expt/2} {

        # #put zeros into old file (in case watchdog hanges, won't slew)
        # exec echo 0.0 0.0 $target_name 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt
        # #save time the file was modified
        # set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

	#take exposure
	puts [concat "Taking exposure for" $expt "seconds"]
        go etime=$expt nexp=1 title=$target_name {comment1=} {comment2=}

        # #put zeros into old file (in case watchdog hanges, won't slew)
        exec echo 0.0 0.0 $target_name 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

        #save time the last file was modified immediately after the exposure ends
        set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

        #get current time
	set current_time [ clock seconds ]
	#if time is up, jump out of loop (don't bother calculating offset)
	if {$current_time > $start_time + $total_time - $expt/2 - 10
	} then {break}
 
	# #read output from PINES_watchdog when the file updates
	set data [PINES_watchdog_get $start_file_time]
	set d_ra [expr [lindex $data 0]]
	set d_dec [expr [lindex $data 1]]
	set target_name [lindex $data 2]
	set fwhm [lindex $data 3]
	set d_ra_median [lindex $data 4]
	set d_dec_median [lindex $data 5]

	set separation [expr sqrt(pow($d_ra,2) + pow($d_dec,2))]

	#If the separation is within the lower and upper limits, then slew the telescope
	if {$separation > $lower_limit && $separation < $upper_limit
	} then {
	    set integrated_ra_error [expr $integrated_ra_error + $d_ra]
	    set integrated_dec_error [expr $integrated_dec_error + $d_dec]	    
	    puts [concat "Moving "  [expr $prop * $d_ra + $integ * $integrated_ra_error] "," [expr $prop * $d_dec + $integ * $integrated_dec_error] " arcsec..."]
	    rmove d_ra=[expr $prop * $d_ra + $integ * $integrated_ra_error] d_dec=[expr $prop * $d_dec + $integ * $integrated_dec_error]
	} else {puts [concat "Offset of"  [expr $d_ra] "," [expr $d_dec] "outside of allowable ranges for guiding."]}

	#if there was an RA slip, correct the full amount without the prop of integ components
	if {[expr abs($d_ra)] > $upper_limit && [expr $d_ra] > -90 && [expr $d_ra] < -40  && [expr abs($d_dec_median)] < $upper_limit
	} then {
	    
	    puts [concat "RA slip detected, moving "  [expr $d_ra ] "," [expr $d_dec] " arcsec..."]
	    rmove d_ra=[expr $d_ra] d_dec=[expr $d_dec]
	}

        #get current time
	set current_time [ clock seconds ]

	#print out time left
        set time_left [expr $start_time + $total_time - $current_time ]
        puts [concat "Time left:" $time_left "seconds..."]

	#increment index
	incr index
    }

    #turn display, telescope updates and logging back on
    display_on
    set tmsg [ catch { Tele_Gui_ON } result ]
    Logging_ON


    puts "PINES_guide COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 9} {incr i} {
	puts \a\a
	after 300
    }

}

proc PINES_JHK {expt_J nexp_J expt_H nexp_H expt_K nexp_K total_time} {

    puts "STARTING PINES_JHK..."

    set pines_date [PINES_get_date]

    # set target name to be whatever is in image_shift currently
    set fid [open /mimir/data/obs72/$pines_date/image_shift.txt r]
    set text [gets $fid]
    close $fid
    set list [split $text " "]
    set target_name [lindex $list 2]
 
    # #put zeros into old file
    # exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

    # #save time the file was modified
    # set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

    # puts "Taking a test exposure to get target name..."	    
    # test etime=1 nexp=1 {comment1=} {comment2=}

    # #read output from PINES_watchdog
    # set data [PINES_watchdog_get $start_file_time]
    # set d_ra [expr [lindex $data 0]]
    # set d_dec [expr [lindex $data 1]]
    # set target_name [lindex $data 2]
    # set fwhm [lindex $data 3]

    puts [concat "Target name is" $target_name "."]

    #set limits for guiding corrections in arcseconds
    set upper_limit 10.0
    set lower_limit 0.5

    #set guiding coefficients
    set prop 1.0
    set integ 0.1
    set integrated_ra_error 0.0
    set integrated_dec_error 0.0
    
    #get start time
    set start_time [ clock seconds ]

    #puts [concat "Taking" $expt "second exposures for" $total_time "seconds..."]

    #set index to 0
    set index 0

    #get current time for while loop
    set current_time [ clock seconds ]

    #turn off display, telescope updates and logging to save time
    display_off
    set tmsg [ catch { Tele_Gui_OFF } ]
    Logging_OFF


    #start while loop, continue until time elapses and a full cycle completes
    while {($current_time < $start_time + $total_time) || ($index % ($nexp_J+$nexp_H+$nexp_K) != 0) } {

        # #put zeros into old file (in case watchdog hanges, won't slew)
        # exec echo 0.0 0.0 $target_name 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt
        # #save time the file was modified
        # set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

	# if {[expr $index % 3] == 0} then {set_J}
	# if {[expr $index % 3] == 1} then {set_H}
	# if {[expr $index % 3] == 2} then {set_K}

	# if {[expr $index/$nexp % 3] == 0} then {set_J}
	# if {[expr $index/$nexp % 3] == 1} then {set_H}
	# if {[expr $index/$nexp % 3] == 2} then {set_K}

	if {[expr $index % ($nexp_J+$nexp_H+$nexp_K)] == 0} then {
	    set_J
	    set expt [expr $expt_J]
	    set nexps [expr $nexp_J] } 
	if {[expr $index % ($nexp_J+$nexp_H+$nexp_K)] == $nexp_J} then {
	    set_H
	    set expt [expr $expt_H] 
	    set nexps [expr $nexp_H] }
	if {[expr $index % ($nexp_J+$nexp_H+$nexp_K)] == [expr ($nexp_J+$nexp_H)]} then {
	    set_K
	    set expt [expr $expt_K] 
	    set nexps [expr $nexp_K] }
        # if there is a problem here, check expr, formatting may not be correct



	#take exposure
	puts [concat "Taking exposure for" $expt "seconds"]
        go etime=$expt nexp=nexps title=$target_name {comment1=} {comment2=}
#set nexp for each band in chunk of text above and then set nexp=nexpbeforeguide. If you make this change, you need to change incr index (last line) to be something like set index [expr ($index+10)]
        # #put zeros into old file (in case watchdog hanges, won't slew)
        exec echo 0.0 0.0 $target_name 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

        #save time the last file was modified immediately after the exposure ends
        set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

        #get current time
	set current_time [ clock seconds ]
 
	# #read output from PINES_watchdog when the file updates
	set data [PINES_watchdog_get $start_file_time]
	set d_ra [expr [lindex $data 0]]
	set d_dec [expr [lindex $data 1]]
	set target_name [lindex $data 2]
	set fwhm [lindex $data 3]
	set d_ra_median [lindex $data 4]
	set d_dec_median [lindex $data 5]

	set separation [expr sqrt(pow($d_ra,2) + pow($d_dec,2))]

	#If the separation is within the lower and upper limits, then slew the telescope
	if {$separation > $lower_limit && $separation < $upper_limit
	} then {
	    set integrated_ra_error [expr $integrated_ra_error + $d_ra]
	    set integrated_dec_error [expr $integrated_dec_error + $d_dec]	    
	    puts [concat "Moving "  [expr $prop * $d_ra + $integ * $integrated_ra_error] "," [expr $prop * $d_dec + $integ * $integrated_dec_error] " arcsec..."]
	    rmove d_ra=[expr $prop * $d_ra + $integ * $integrated_ra_error] d_dec=[expr $prop * $d_dec + $integ * $integrated_dec_error]
	} else {puts [concat "Offset of"  [expr $d_ra] "," [expr $d_dec] "outside of allowable ranges for guiding."]}

	#if there was an RA slip, correct the full amount without the prop of integ components
	if {[expr abs($d_ra)] > $upper_limit && [expr $d_ra] > -90 && [expr $d_ra] < -40  && [expr abs($d_dec_median)] < $upper_limit
	} then {
	    
	    puts [concat "RA slip detected, moving "  [expr $d_ra ] "," [expr $d_dec] " arcsec..."]
	    rmove d_ra=[expr $d_ra] d_dec=[expr $d_dec]
	}

        #get current time
	set current_time [ clock seconds ]

	#print out time left
        set time_left [expr $start_time + $total_time - $current_time ]
        puts [concat "Time left:" $time_left "seconds..."]

	#increment index
	set index [expr ($index+1)]
    }

    puts "Setting filter back to J"
    set_J

    #turn display, telescope updates and logging back on
    display_on
    set tmsg [ catch { Tele_Gui_ON } result ]
    Logging_ON


    puts "PINES_JHK COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 9} {incr i} {
	puts \a\a
	after 300
    }

}



proc PINES_dither {expt total_time} {

    puts "STARTING PINES_dither..."

    set pines_date [PINES_get_date]

    # set target name to be whatever is in image_shift currently
    set fid [open /mimir/data/obs72/$pines_date/image_shift.txt r]
    set text [gets $fid]
    close $fid
    set list [split $text " "]
    set target_name [lindex $list 2]
 
    # #put zeros into old file
    # exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

    # #save time the file was modified
    # set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

    # puts "Taking a test exposure to get target name..."	    
    # test etime=1 nexp=1 {comment1=} {comment2=}

    # #read output from PINES_watchdog
    # set data [PINES_watchdog_get $start_file_time]
    # set d_ra [expr [lindex $data 0]]
    # set d_dec [expr [lindex $data 1]]
    # set target_name [lindex $data 2]
    # set fwhm [lindex $data 3]

    puts [concat "Target name is" $target_name "."]

    #set limits for guiding corrections in arcseconds
    set upper_limit 15.0
    set lower_limit 0.5

    #set guiding coefficients
    #prop of 1.0 is full correction
    set prop 0.9
    #turn off integral term for now (set to zero)
    set integ 0.1 
    set integrated_ra_error 0.0
    set integrated_dec_error 0.0
    
    #get start time
    set start_time [ clock seconds ]

    puts [concat "Taking" $expt "second exposures for" $total_time "seconds..."]

    #set index to 1
    set index 1

    #get current time for while loop
    set current_time [ clock seconds ]

    #relative positions for dither, assume we start at the master location
    set new_ra_pos 0.0
    set new_dec_pos 0.0

    #corrections
    set ra_correction 0.0
    set dec_correction 0.0


    #start while loop
    while {$current_time < $start_time + $total_time - 2 * $expt} {

	puts " "
	puts [concat "Current time is " $current_time]

	# Make a for loop for each dither position
	
	for {set i 0} {$i < 4} {incr i} {

	    set old_ra_pos $new_ra_pos	    
	    set old_dec_pos $new_dec_pos
	    
	    #_pos indicates a position relative to the master image
	    switch $i {
		0 {
		    set new_ra_pos -7.5
		    set new_dec_pos -7.5
		    set position "Dither Position A"
		}
		1 {
		    set new_ra_pos -7.5
		    set new_dec_pos +7.5
		    set position "Dither Position B"
		}

		2 {
		    set new_ra_pos +7.5
		    set new_dec_pos +7.5
		    set position "Dither Position C"

		}
		3 {
		    set new_ra_pos +7.5
		    set new_dec_pos -7.5
		    set position "Dither Position D"

		}
		default {
		    set new_ra_pos 0
		    set new_dec_pos 0
		    set position "Error"

		}
	    }

	    puts " "
	    puts [concat "Current time is " $current_time]

	    #Move telescope to position
	    puts [concat "Moving to " $position ", d_ra = " [expr $new_ra_pos - $old_ra_pos + $ra_correction] ", d_dec = " [expr $new_dec_pos - $old_dec_pos + $dec_correction]]
	    rmove d_ra=[expr $new_ra_pos - $old_ra_pos + $ra_correction] d_dec=[expr $new_dec_pos - $old_dec_pos + $dec_correction]
	
	    #take exposure
	    puts [concat "Taking exposure for" $expt " at " $position]
	    go etime=$expt nexp=1 title=$target_name comment1=$position {comment2=}

	    # while watchdog works, put zeros into old file (in case watchdog hanges, won't slew)
	    exec echo 0.0 0.0 $target_name 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

	    #save time the last file was modified immediately after the exposure ends
	    set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]
	    
	    # read output from PINES_watchdog when the file is modified
	    set data [PINES_watchdog_get $start_file_time]
	    set d_ra [expr [lindex $data 0]]
	    set d_dec [expr [lindex $data 1]]
	    set target_name [lindex $data 2]
	    set fwhm [lindex $data 3]
	    set d_ra_median [lindex $data 4]
	    set d_dec_median [lindex $data 5]

	    puts [concat "Watchdog suggests: d_ra = " $d_ra ", d_dec = " $d_dec]

	    #add the dither positions to the suggested motion (which would otherwise CORRECT for the dither motion -> bad)
	    set d_ra [expr $d_ra + $new_ra_pos]
	    set d_dec [expr $d_dec + $new_dec_pos]

	    puts [concat "Suggested offsets minus dither: d_ra = " $d_ra ", d_dec = " $d_dec]
	    
	    set separation [expr sqrt(pow($d_ra,2) + pow($d_dec,2))]

	    #If the separation is within the lower and upper limits, then slew the telescope
	    if {$separation > $lower_limit && $separation < $upper_limit
	    } then {
		set integrated_ra_error [expr $integrated_ra_error + $d_ra]
		set integrated_dec_error [expr $integrated_dec_error + $d_dec]
		set ra_correction [expr $prop * $d_ra + $integ * $integrated_ra_error]
		set dec_correction [expr $prop * $d_dec + $integ * $integrated_dec_error]
		puts [concat "Calculated correction: d_ra = " $ra_correction ", d_dec = " $dec_correction]		
	    } else {
		puts [concat "Offset of"  [expr $d_ra] "," [expr $d_dec] "outside of allowable ranges for guiding."]
		set ra_correction 0.0
		set dec_correction 0.0
	    }

	    #if there was an RA slip, correct the full amount without the prop or integ components
	    if {[expr abs($d_ra)] > $upper_limit && [expr $d_ra] > -90 && [expr $d_ra] < -40  && [expr abs($d_dec)] < $upper_limit
	    } then {
		
		puts [concat "RA slip detected, moving "  [expr $d_ra ] "," [expr $d_dec] " arcsec..."]
		rmove d_ra=[expr $d_ra] d_dec=[expr $d_dec]
	    }

	}       
	    
        #get current time
	set current_time [ clock seconds ]

	#print out time left
        set time_left [expr $start_time + $total_time - 2*$expt - $current_time ]
        puts [concat "Time left:" $time_left "seconds..."]

	#increment index
	incr index


    }

    puts " "
    puts "PINES_dither COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

}



#program to calculate LST using computer system time and assuming mountain standard (GMT-7)
proc PINES_lst {} {

    #Time zone (hours east of Greenwich, -6 for MST)
    set tz {-7}
    set lng -111.53592

    set systemTime [clock seconds]    
    set yr [scan [clock format $systemTime -format %Y] %d]
    set mn [scan [clock format $systemTime -format %m] %d]
    set day [scan [clock format $systemTime -format %d] %d]
    set hr [scan [clock format $systemTime -format %H] %d]
    set min [scan [clock format $systemTime -format %M] %d]
    set sec [scan [clock format $systemTime -format %S] %d]

    set hr [expr {$hr} + {$min} / 60.0 + {$sec} / 3600.0 - $tz]
    #set hr [expr [expr [scan $hr %d]] + [expr [scan $min %d]] / 60.0 + [expr [scan $sec %d]] / 3600.0 - $tz]  

    if {$mn == 1 || $mn == 2
    } then {set L -1
    } else {set L 0}

    set julian [expr $day - 32075 + 1461*($yr+4800+$L)/4 + 367*($mn - 2 - $L*12)/12 - 3*(($yr + 4900 + $L)/100)/4]
    #set julian [expr [scan $day %d] - 32075 + 1461*($yr+4800+$L)/4 + 367*([scan $mn %d] - 2 - $L*12)/12 - 3*(($yr + 4900 + $L)/100)/4]

    set jd [expr $julian + ($hr / 24.0) - 0.5]
    #puts [concat "JD=" $jd]
    
    set c {280.46061837 360.98564736629 0.000387933 38710000.0}

    set jd2000 2451545.0

    set t0 [expr $jd - $jd2000]

    set t [expr $t0/36525]

    set theta [expr [lindex $c 0] + ([lindex $c 1] * $t0) + pow($t,2)*([lindex $c 2] - $t / [lindex $c 3])]

    set angle [expr ($theta + $lng) / 15.0]

    set lst [expr ([expr int($angle * 1000)] % 24000)/1000.0]
    
    puts [concat "LST=" $lst]

    return $lst
}

#program to slew to a target in a group.txt file
proc PINES_slew { file_name target_num } { 

     puts "STARTING PINES_slew..."

     set pines_date [PINES_get_date]

     #set file_name "group.txt"

     puts [concat "Reading group coordinates from" /mimir/data/obs72/$pines_date/$file_name]
     set fid [open /mimir/data/obs72/$pines_date/$file_name r]
    
    set index 0
    while { [gets $fid line] >= 0 } {
	set list [split $line " "]

	lappend target_list [lindex $list 0]
	lappend RAhrs_list [lindex $list 1]
	lappend RAmin_list [lindex $list 2]
	lappend RAsec_list [lindex $list 3]
	lappend DECdeg_list [lindex $list 4]
	lappend DECmin_list [lindex $list 5]
	lappend DECsec_list [lindex $list 6]
	lappend expt_list [lindex $list 7]	
	lappend index_list $index

	incr index
    }
    close $fid



    set DECdeg [lindex $DECdeg_list $target_num]
    set DECmin [lindex $DECmin_list $target_num]
    set DECsec [lindex $DECsec_list $target_num]
    
    #make sure the telescope can move to the coordinates
    #covert sexigessimal to decimal
    set ra_deg [expr [expr [scan $RAhrs %d]] + [expr [scan $RAmin %d]] / 60.0 + [expr [scan $RAsec %d]] / 3600.0]
    #for dec, only use degree to avoid sign issues with min and sec
    set dec_deg [expr [scan $DECdeg %d]]

    #calculate hour angle
    set ha [expr [PINES_lst] - $ra_deg]

    if {[expr $ha] < -12}
    then { set ha [expr $ha + 24.0] }

    if {[expr $ha] > 12} 
    then {set ha [expr $ha - 24.0]}

    if {[expr abs($ha)] > 4.25}
    then {puts [concat "Hour Angle of" $ha "passed safe limits. Move not executed."]}
    elseif {$ha < 0.0 && $dec_deg > 45.0}
    then {puts [concat "Target at Hour Angle of" $ha "and Dec of" $dec_deg " degrees, near a restricted position. Move not executed."] }
    else { 
	puts [concat "Moving to" $target "at" $RAhrs $RAmin $RAsec $DECdeg $DECmin $DECsec ", HA =" $ha "hours..."]
	move ra=$RAhrs:$RAmin:$RAsec dec=$DECdeg:$DECmin:$DECsec
    }

   puts "PINES_slew COMPLETE."


   for {set i 0} {$i < 3} {incr i} {
       puts \a\a
       after 300
   }


 }


#program to cycle through a group of targets
proc PINES_group {file_name start_num peakup_expt time_per_star} {

    puts "STARTING PINES_group..."

    set pines_date [PINES_get_date]

    #set group_file_name "group.txt"

    puts [concat "Reading group coordinates from" /mimir/data/obs72/$pines_date/$file_name]
    set fid [open /mimir/data/obs72/$pines_date/$file_name r]
    
    set index 0
    while { [gets $fid line] >= 0 } {
	set list [split $line " "]

	lappend target_list [lindex $list 0]	
	lappend RAhrs_list [lindex $list 1]
	lappend RAmin_list [lindex $list 2]
	lappend RAsec_list [lindex $list 3]
	lappend DECdeg_list [lindex $list 4]
	lappend DECmin_list [lindex $list 5]
	lappend DECsec_list [lindex $list 6]
	lappend expt_list [lindex $list 7]	
	lappend index_list $index

	incr index
    }
    close $fid

    #Start loop through targets in group
    foreach index $index_list {

	if {$index < $start_num} {continue}

	set target [lindex $target_list $index]
	set expt [lindex $expt_list $index]
	
	set RAhrs [lindex $RAhrs_list $index]
	set RAmin [lindex $RAmin_list $index]
	set RAsec [lindex $RAsec_list $index]

	set DECdeg [lindex $DECdeg_list $index]
	set DECmin [lindex $DECmin_list $index]
	set DECsec [lindex $DECsec_list $index]
	
	#make sure the telescope can move to the coordinates
	#covert sexigessimal to decimal
	set ra_deg [expr [expr [scan $RAhrs %d]] + [expr [scan $RAmin %d]] / 60.0 + [expr [scan $RAsec %d]] / 3600.0]
	#for dec, only use degree to avoid sign issues with min and sec
	set dec_deg [expr [scan $DECdeg %d]]

	#calculate hour angle
	set ha [expr [PINES_lst] - $ra_deg]

	if {[expr $ha] < -12
	} then { set ha [expr $ha + 24.0] }

	if {[expr $ha] > 12
	} then { set ha [expr $ha - 24.0] }

	if {[expr abs($ha)] > 4.25
	} then { puts [concat "Hour Angle of" $ha "greater than 4.25. Move not executed."]
	} elseif { $ha < 0.0 && $dec_deg > 45.0
	       } then { puts [concat "Target at Hour Angle of" $ha "and Dec of" $dec_deg " degrees, near a restricted position. Move not executed."]
	} else { puts [concat "Moving to" $target "at" $RAhrs $RAmin $RAsec $DECdeg $DECmin $DECsec ", HA =" $ha "hours..."]
	    
	    move ra=$RAhrs:$RAmin:$RAsec dec=$DECdeg:$DECmin:$DECsec

	    #Give tele 5 seconds to settle before running peakup
	    after 5000

	    puts "Running PINES_peakup."
	    set successful_peakup [PINES_peakup $peakup_expt]

	    if {$successful_peakup == -1
	    } then {puts "PINES_peakup unsuccessful. STOPPING loop."
		break
	    } else {puts "PINES_peakup successful. Running PINES_guide."}
    
	    PINES_guide $expt $time_per_star
	    
	}
    }
    
   

    puts "PINES_group COMPLETE."

    #Three quarks for muster mark.    
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }
    
}



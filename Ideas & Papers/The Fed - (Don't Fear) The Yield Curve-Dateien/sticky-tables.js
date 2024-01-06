if (!Array.prototype.fill) {
  Array.prototype.fill = function(value) {

    // Steps 1-2.
    if (this == null) {
      throw new TypeError('this is null or not defined');
    }

    var O = Object(this);

    // Steps 3-5.
    var len = O.length >>> 0;

    // Steps 6-7.
    var start = arguments[1];
    var relativeStart = start >> 0;

    // Step 8.
    var k = relativeStart < 0 ?
      Math.max(len + relativeStart, 0) :
      Math.min(relativeStart, len);

    // Steps 9-10.
    var end = arguments[2];
    var relativeEnd = end === undefined ?
      len : end >> 0;

    // Step 11.
    var final = relativeEnd < 0 ?
      Math.max(len + relativeEnd, 0) :
      Math.min(relativeEnd, len);

    // Step 12.
    while (k < final) {
      O[k] = value;
      k++;
    }

    // Step 13.
    return O;
  };
}
pubwebApp.directive('stickyColumns', [function() {
    return {
        scope: true,
        link: function(scope, el, attr) {
            scope.stickyColumns = {
                table: el[0],
                containerOffsetLeft: 0,
                columnCount: 0,
                scrollX: 0,
                stickyColumnCells: [],
                stickyColumnCount: 1,
                totalStickyWidth: 0,
                scrollContainer: null,
                resizeHandler: function(){
                    getScrollContainerAndOffset(scope.stickyColumns.table,scope);
                    getTotalStickyWidth();
                },

                scrollHandler: function() {

                    var positionX;
                    var positionY = 0;

                    // if the scrollPosition is greater than the table position
                    if(scope.stickyColumns.scrollLeft > scope.stickyColumns.containerOffsetLeft){
                        positionX = Math.min(scope.stickyColumns.table.scrollWidth - scope.stickyColumns.totalStickyWidth, scope.stickyColumns.scrollLeft - scope.stickyColumns.containerOffsetLeft);
                    }else{
                        positionX = 0;
                    }

                    angular.forEach(scope.stickyColumns.stickyColumnCells, function(cell) {
                        if(cell.classList.contains('sticky-row-cell') && cell.style.transform.indexOf('') >= 0){
                            positionY = cell.style.transform.split(',')[1];
                        }else{
                            positionY = 0;
                        }
                        if(cell.classList.contains('sticky-column-cell__fullspan')){
                            var originalPaddingLeft = parseInt(cell.getAttribute('data-padding-left'));
                            cell.style.paddingLeft = originalPaddingLeft + positionX + 'px';
                        }else{
                            cell.style.transform = 'translate3d('+positionX+'px,'+positionY+',0)';
                        }
                    });

                }
            }

            scope.stickyColumns.table.classList.add('sticky-table');

            // if a sticky columns count attribute is set, use it
            if (attr.stickyColumns) {
                scope.stickyColumns.stickyColumnCount = parseInt(attr.stickyColumns);
            }

            // don't do anything else if this isn't a positive integer
            if (scope.stickyColumns.stickyColumnCount < 1) {
                return;
            }

            var trs = el.find('tr');
            var child = null;
            var colspan = 0;
            var rowspan = 0;
            var rowIndex = 0;

            // how many columns does the table have?
            angular.forEach(trs[0].children,function(cell){
                if(cell.hasAttribute('colspan')){
                    scope.stickyColumns.columnCount += parseInt(cell.getAttribute('colspan'));
                }else{
                    scope.stickyColumns.columnCount++;
                }
            });


            /******* This is tricky, pay attention **********/
            // rowspansByColumn is an array to keep track of rowspans indexed by column
            // when a new rowspan is encountered, set the indexed counter to the rowspan value
            // then for each subsequent row, decrement the value back down to 0

            // initialize an array with length or stickyColumnCount and preset them all to 0
            var rowspansByColumn = Array(scope.stickyColumns.stickyColumnCount).fill(0);

            // for each row, look for the first N columns to make sticky (unless working on a rowspan)
            angular.forEach(trs, function(row) {

                if(row.children.length === 0){
                    return;
                }
                
                // columnIndex can differ from cellIndex because of rowspans
                var columnIndex = 0; // columnIndex is a counter for the columns that SHOULD be there
                var cellIndex = 0; // cellIndex is an index into the actual cells of the tr
                
                while (columnIndex < scope.stickyColumns.stickyColumnCount) {

                    cell = row.children[cellIndex];

                    // reset rowspan accounting
                    rowspan = 0;

                    // skip over columns that still have some span left
                    while(rowspansByColumn[columnIndex]){
                        rowspansByColumn[columnIndex]--;
                        columnIndex++;
                    }

                    if(columnIndex >= scope.stickyColumns.stickyColumnCount){
                        continue;
                    }

                    scope.stickyColumns.stickyColumnCells.push(cell);
                    cell.classList.add('sticky');
                    cell.classList.add('sticky-column-cell');

                    // reset the transform
                    cell.style.transform = 'none';

                    // if this cell has a rowspan, set it's value in rowspansByColumn
                    if (cell.hasAttribute('rowspan')) {
                        rowspan = parseInt(cell.getAttribute('rowspan'));
                        if (rowspan > 1) {
                            rowspansByColumn[columnIndex] = rowspan-1; // the "-1" because the current row is currently being handled
                        }
                    }

                    // if this cell has a colspan > 1
                    if (cell.hasAttribute('colspan')) {
                        colspan = parseInt(cell.getAttribute('colspan'));
                        if (colspan > 1) {

                            // if it's a full-span or covers the rest of the row
                            if(colspan === scope.stickyColumns.columnCount || (columnIndex + colspan) === scope.stickyColumns.columnCount){
                                var style =  window.getComputedStyle(cell);
                                cell.classList.add('sticky-column-cell__fullspan');
                                cell.setAttribute('data-padding-left',parseInt(style.getPropertyValue('padding-left')));
                            }

                            // if it was also a rowspan, add entries for all of the column indexes in rowspansByColumn
                            for(var j = 1; rowspan > 1 && j < colspan; j++){
                                rowspansByColumn[columnIndex+j] = rowspan-1; // the "-1" because the current row is currently being handled
                            }
                            // cellIndex = cellIndex + colspan - 1;
                            columnIndex = columnIndex + colspan - 1;
                        }
                    }
                    columnIndex++;
                    cellIndex++;
                }
                rowIndex++;
            });

            getScrollContainerAndOffset(scope.stickyColumns.table,scope);
            getTotalStickyWidth();

            window.addEventListener('resize', function(e) {
                window.requestAnimationFrame(scope.stickyColumns.resizeHandler);
            });

            function scrollListener(e) {
                if (scope.stickyColumns.scrollContainer.scrollLeft !== scope.stickyColumns.scrollLeft) {
                    scope.stickyColumns.scrollLeft = scope.stickyColumns.scrollContainer.scrollLeft;
                    window.requestAnimationFrame(scope.stickyColumns.scrollHandler);
                }
            }

            function getScrollContainerAndOffset(el,scope){

                scope.stickyColumns.containerOffsetLeft = el.clientLeft;

                if(scope.stickyColumns.scrollContainer != null){
                    scope.stickyColumns.scrollContainer.removeEventListener('scroll',scrollListener);
                    scope.stickyColumns.scrollContainer = null;
                }

                // go up through the dom and figure out the real offset into the page.
                while(el.parentNode){
                    el = el.parentNode;
                    scope.stickyColumns.containerOffsetLeft += el.clientLeft;
                    if(el.clientWidth < el.scrollWidth){
                        scope.stickyColumns.scrollContainer = el;
                        scope.stickyColumns.scrollContainer.addEventListener('scroll',scrollListener);
                        break;
                    }
                }
            }

            function getTotalStickyWidth(){
                var cellBoundingRect;

                // default to extremes to find the limits of the cells
                var minLeft = Number.MAX_SAFE_INTEGER;
                var maxRight = Number.MIN_SAFE_INTEGER;

                // loop through all the cells and see what the sticky offset max should be
                angular.forEach(scope.stickyColumns.stickyColumnCells, function(cell) {

                    // if it's a full-colspan cell, don't count it for total width
                    if(cell.classList.contains('sticky-column-cell__fullspan')){
                        return;
                    }

                    cellBoundingRect = cell.getBoundingClientRect();
                    minLeft = Math.min(minLeft,cellBoundingRect.left);
                    maxRight = Math.max(maxRight,cellBoundingRect.right);
                });
                scope.stickyColumns.totalStickyWidth = maxRight - minLeft;
            }

        }
    }
}]);

pubwebApp.directive('stickyRows', [function() {

    return {
        scope: true,
        link: function(scope, el, attr) {

            scope.stickyRows = {
                table: el[0],
                scrollY: 0,
                containerOffsetTop: 0,
                stickyRowCells: [],
                stickyRowCount: 1,
                totalStickyHeight: 0,
                otherStickyEls: [],
                scrollContainer: null,
                resizeHandler: function(){
                    getScrollContainerAndOffset(scope.stickyRows.table,scope);
                    getTotalStickyHeight();
                },

                scrollHandler: function() {
                
                    var positionY = 0;
                    var positionX = 0;
                    var style;
                    var tableRect = scope.stickyRows.table.getBoundingClientRect();

                    if(scope.stickyRows.scrollContainer === window){
                        // height of other sticky items
                        var additionalOffset = 0;

                        // if a otherStickyEls has been specified and it is currently "postion: fixed", see where its bottom is
                        scope.stickyRows.otherStickyEls.forEach(function(stuckEl){
                            style =  window.getComputedStyle(stuckEl);
                            if(style.getPropertyValue('position') === 'fixed'){
                                additionalOffset = Math.max(additionalOffset,stuckEl.getBoundingClientRect().bottom);
                            }
                        });

                        //
                        if(tableRect.top < additionalOffset){
                            positionY = Math.min(tableRect.height - scope.stickyRows.totalStickyHeight, additionalOffset - tableRect.top);
                        }else{
                            positionY = 0;
                        }
                    }else if(scope.stickyRows.scrollContainer.scrollTop > scope.stickyRows.containerOffsetTop){
                        positionY = scope.stickyRows.scrollContainer.scrollTop - scope.stickyRows.containerOffsetTop;
                    }

                    angular.forEach(scope.stickyRows.stickyRowCells, function(cell) {

                        if(cell.classList.contains('sticky-column-cell') && cell.style.transform.indexOf('translate3d') >= 0){
                            positionX = cell.style.transform.split(/[(,]/)[1];
                        }else{
                            positionX = 0;
                        }
                        cell.style.transform = 'translate3d('+positionX+','+positionY+'px,0)';
                    });
                },

            }

            scope.stickyRows.table.classList.add('sticky-table');

            // if a sticky rows count attribute is set, use it
            if (attr.stickyRows) {
                scope.stickyRows.stickyRowCount = parseInt(attr.stickyRows);
            }

            // don't do anything else if this isn't a positive integer
            if (scope.stickyRows.stickyRowCount < 1) {
                return;
            }

            scope.stickyRows.otherStickyEls = getOtherStickyEls();

            var trs = el.find('tr');
            var tr;

            // for each row, work on all children
            for (var i = 0; i < scope.stickyRows.stickyRowCount; i++) {
                tr = trs[i];

                angular.forEach(tr.children,function(cell){
                    scope.stickyRows.stickyRowCells.push(cell);
                    cell.classList.add('sticky');
                    cell.classList.add('sticky-row-cell');
    
                    // reset the transform
                    cell.style.transform = 'none';

                });
            }


            getTotalStickyHeight();
            getScrollContainerAndOffset(scope.stickyRows.table,scope);

            window.addEventListener('resize', function(e) {
                window.requestAnimationFrame(scope.stickyRows.resizeHandler);
            });

            function scrollListener(e) {

                // scroll event could scrollX. limit to scrollY changes
                if (scope.stickyRows.scrollContainer === window && window.pageYOffset !== scope.stickyRows.scrollTop) {
                    scope.stickyRows.scrollTop = window.pageYOffset;
                    window.requestAnimationFrame(function(){
                        scope.stickyRows.scrollHandler();
                    });
                }else if(scope.stickyRows.scrollContainer !== window && scope.stickyRows.scrollContainer.scrollTop !== scope.stickyRows.scrollTop){
                    scope.stickyRows.scrollTop = scope.stickyRows.scrollContainer.scrollTop;
                    window.requestAnimationFrame(function(){
                        scope.stickyRows.scrollHandler();
                    });
                }
            }


            function getScrollContainerAndOffset(el,scope){

                if(scope.stickyRows.scrollContainer != null){
                    scope.stickyRows.scrollContainer.removeEventListener('scroll',scrollListener);
                    scope.stickyRows.scrollContainer = null;
                }
                
                var parent = el.parentNode;
                if(parent.scrollHeight - parent.clientHeight > 20){
                    scope.stickyRows.scrollContainer = parent;
                    var prevSiblingEl = el.previousElementSibling;
                    while(prevSiblingEl){
                        scope.stickyRows.containerOffsetTop += prevSiblingEl.offsetHeight;
                        prevSiblingEl = prevSiblingEl.previousElementSibling;
                    }
                }else{
                    scope.stickyRows.scrollContainer = window;
                }
                scope.stickyRows.scrollContainer.addEventListener('scroll',scrollListener);
            }

            function getTotalStickyHeight(){
                var cellBoundingRect;

                // TODO: account for scrolling in the table?

                // default to extremes to find the limits of the cells
                var minTop = Number.MAX_SAFE_INTEGER;
                var maxBottom = Number.MIN_SAFE_INTEGER;

                // loop through all the cells and see what the sticky offset max should be
                angular.forEach(scope.stickyRows.stickyRowCells, function(cell) {
                    cellBoundingRect = cell.getBoundingClientRect();
                    minTop = Math.min(minTop,cellBoundingRect.top);
                    maxBottom = Math.max(maxBottom,cellBoundingRect.bottom);
                });
                scope.stickyRows.totalStickyHeight = maxBottom - minTop;
            }

            function getOtherStickyEls(){
                var stickyEls = [];
                ['nav-primary','t1_nav','t4_nav--horizontal','t2__offcanvas'].forEach(function(className){
                    var els = document.getElementsByClassName(className);
                    if(els.length){
                        stickyEls.push(els[0]);
                    }
                });
                return stickyEls;
            }
        }
    }
}]);

pubwebApp.controller('t4NavController', function($scope, $element) {
        // $scope.modalTitle = "";
        $scope.$element = $element;

    })

    .directive("expandableT4", function($compile) {

        return function(scope, el, attrs) {

            // start with 80 because padding-bottom
            scope.fullHeight = 80;

            scope.navList = {
                expanderIcon: "icon-T4-expand",
                expanderLabel: "Show More",
                expanded: false
            };

            scope.toggleNavExpansion = function($event) {
                scope.navList.expanded = !scope.navList.expanded;
                if (scope.fullHeight === 80) {
                    for (var i = 0; i < el.children().length; i++) {
                        scope.fullHeight += el.children()[i].offsetHeight;
                    }
                }
                if (scope.navList.expanded) {
                    // console.log('expanding to ' + scope.fullHeight);
                    scope.$element[0].style.maxHeight = scope.fullHeight + "px";
                    scope.navList.expanderLabel = 'Show Less';
                    scope.navList.expanderIcon = 'icon-T4-collapse';
                } else {
                    // console.log('collapsing to 200');
                    scope.$element[0].style.maxHeight = "200px";
                    scope.navList.expanderLabel = 'Show More';
                    scope.navList.expanderIcon = 'icon-T4-expand';
                }
            };

            var html = '<li class="nav-expander" data-ng-click="toggleNavExpansion($event)">' + '<span class="icon icon__sm icon--centered {{navList.expanderIcon}}"></span>' + '<div>{{navList.expanderLabel}}<div>' + '</li>';
            if (el.children().length > 5) {


                var content = $compile(html)(scope);
                el.append(content);
                el[0].style.maxHeight = "200px";
                document.querySelector("style").textContent += "@media (max-width: 767px) {#t4_nav.t4_nav--horizontal ul {  padding-bottom:80px;}}";

            }
        };
    });
